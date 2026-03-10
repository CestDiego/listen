"""
Skill expert HTTP server — parallel inference via worker processes.

Architecture:
  - One HTTP process handles all requests (single-threaded, no Metal usage)
  - One worker process per skill, each with its own Metal context
  - Workers are pre-warmed at startup (model loaded, first inference done)
  - Requests are dispatched to workers via multiprocessing queues
  - Parallel classification: all workers run simultaneously → ~2× speedup

Endpoints:
  GET  /health                → {"status": "ok", "skills": [...], ...}
  POST /v1/classify/{skill}   → single skill classification
  POST /v1/classify           → all skills in parallel (per-expert timing)

Usage:
  uv run python -m experts.serve --port 8234
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from typing import Any

from .config import BASE_MODEL, MODELS_DIR, SKILLS


# ── Worker process ─────────────────────────────────────────────────

def _worker_main(
    skill_name: str,
    request_queue: Any,
    result_queue: Any,
    ready_event: Any,
) -> None:
    """
    Long-lived worker: loads model once, then handles classify requests.
    Each worker runs in its own process with its own Metal context.
    """
    import mlx.core as mx
    from mlx_lm import load, generate as mlx_generate
    from .generate import system_prompt
    from .evaluate import parse_output

    pid = os.getpid()
    adapter_path = MODELS_DIR / skill_name / "adapters.safetensors"

    # Load model in this process (gets its own Metal device context)
    if adapter_path.exists():
        result = load(BASE_MODEL, adapter_path=str(adapter_path))
    else:
        result = load(BASE_MODEL)
    model, tokenizer = result[0], result[1]
    sys_prompt = system_prompt(skill_name)

    def greedy_sampler(logits: Any) -> Any:
        return mx.argmax(logits, axis=-1)

    # Signal ready
    ready_event.set()
    print(f"  worker [{pid}] {skill_name} ready")

    while True:
        msg = request_queue.get()
        if msg is None:  # poison pill → shutdown
            break

        transcript = msg["transcript"]
        request_id = msg["id"]

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": transcript},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        start = time.perf_counter()
        output = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=60,
            sampler=greedy_sampler,
            verbose=False,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        parsed = parse_output(output)
        result_data: dict[str, Any] = {
            "skill": skill_name,
            "match": parsed.get("match", False),
            "status": "ok",
            "latency_ms": latency_ms,
            "pid": pid,
        }
        if parsed.get("match"):
            result_data["action"] = parsed.get("action", "")
            result_data["confidence"] = parsed.get("confidence", 0.0)

        result_queue.put({"id": request_id, **result_data})


# ── Worker pool manager ────────────────────────────────────────────

class WorkerPool:
    """Manages pre-warmed worker processes for parallel classification."""

    def __init__(self, skill_names: list[str]):
        self.skill_names = skill_names
        self.workers: dict[str, dict[str, Any]] = {}
        self._request_counter = 0

    def start(self, timeout: float = 60.0) -> None:
        """Start all workers and wait for them to be ready."""
        for skill in self.skill_names:
            req_q: mp.Queue = mp.Queue()
            res_q: mp.Queue = mp.Queue()
            ready = mp.Event()
            proc = mp.Process(
                target=_worker_main,
                args=(skill, req_q, res_q, ready),
                daemon=True,
            )
            proc.start()
            self.workers[skill] = {
                "process": proc,
                "req": req_q,
                "res": res_q,
                "ready": ready,
            }

        # Wait for all workers to load models
        print(f"  Waiting for {len(self.skill_names)} workers to load models...")
        for skill in self.skill_names:
            if not self.workers[skill]["ready"].wait(timeout=timeout):
                raise TimeoutError(f"Worker {skill} failed to start within {timeout}s")

    def classify_single(self, skill_name: str, transcript: str) -> dict[str, Any]:
        """Classify via a single skill worker."""
        if skill_name not in self.workers:
            return {"skill": skill_name, "match": False, "status": "error",
                    "error": f"Unknown skill: {skill_name}", "latency_ms": 0}

        self._request_counter += 1
        req_id = self._request_counter
        self.workers[skill_name]["req"].put({"transcript": transcript, "id": req_id})
        return self.workers[skill_name]["res"].get(timeout=30)

    def classify_parallel(self, transcript: str,
                          skills: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Classify against all skills IN PARALLEL.
        Each worker runs in its own process with its own Metal context.
        """
        target_skills = skills or self.skill_names
        self._request_counter += 1
        req_id = self._request_counter

        # Dispatch to all workers simultaneously
        for skill in target_skills:
            if skill in self.workers:
                self.workers[skill]["req"].put({"transcript": transcript, "id": req_id})

        # Collect results (order doesn't matter)
        results = []
        for skill in target_skills:
            if skill in self.workers:
                results.append(self.workers[skill]["res"].get(timeout=30))
            else:
                results.append({"skill": skill, "match": False, "status": "error",
                                "error": f"Unknown skill: {skill}", "latency_ms": 0})

        return results

    def shutdown(self) -> None:
        """Stop all workers."""
        for skill in self.skill_names:
            if skill in self.workers:
                self.workers[skill]["req"].put(None)
        for skill in self.skill_names:
            if skill in self.workers:
                self.workers[skill]["process"].join(timeout=5)


# ── HTTP server ────────────────────────────────────────────────────

def run_server(port: int = 8234, skills: list[str] | None = None) -> None:
    """Run the HTTP classification server with parallel worker pool."""
    import http.server
    import socketserver

    active_skills = skills or list(SKILLS.keys())

    # Start worker pool
    pool = WorkerPool(active_skills)
    pool.start()
    print(f"  All {len(active_skills)} workers ready.\n")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                response = {
                    "status": "ok",
                    "skills": active_skills,
                    "base_model": BASE_MODEL,
                    "parallel": True,
                    "workers": len(active_skills),
                    "endpoints": [
                        f"/v1/classify/{s}" for s in active_skills
                    ] + ["/v1/classify"],
                }
                self._send_json(200, response)
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            # Per-skill endpoint
            for skill_name in active_skills:
                if self.path == f"/v1/classify/{skill_name}":
                    self._handle_single(skill_name)
                    return

            # Bulk parallel endpoint
            if self.path == "/v1/classify":
                self._handle_parallel()
                return

            self.send_error(404)

        def _handle_single(self, skill_name: str) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                req = json.loads(body)
                transcript = req.get("transcript", "")

                if not transcript or len(transcript.strip()) < 10:
                    self._send_json(200, {
                        "skill": skill_name, "match": False,
                        "status": "too_short", "latency_ms": 0,
                    })
                    return

                result = pool.classify_single(skill_name, transcript)
                # Remove internal fields
                result.pop("id", None)
                result.pop("pid", None)

                ms = result.get("latency_ms", 0)
                matched = result.get("match", False)
                icon = "+" if matched else "-"
                action = f".{result.get('action', '?')}" if matched else ""
                print(f"  [{ms}ms] {icon} {skill_name}{action} \"{transcript[:50]}\"")

                self._send_json(200, result)

            except Exception as e:
                self._send_json(500, {
                    "skill": skill_name, "match": False,
                    "status": "error", "error": str(e), "latency_ms": 0,
                })

        def _handle_parallel(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                req = json.loads(body)
                transcript = req.get("transcript", "")
                req_skills = req.get("skills", active_skills)

                if not transcript or len(transcript.strip()) < 10:
                    self._send_json(200, {"results": [], "reason": "too short"})
                    return

                wall_start = time.perf_counter()
                results = pool.classify_parallel(transcript, req_skills)
                wall_ms = round((time.perf_counter() - wall_start) * 1000, 1)

                # Clean up internal fields
                for r in results:
                    r.pop("id", None)
                    r.pop("pid", None)

                # Compute expert sum for parallelism metric
                expert_sum_ms = round(
                    sum(r.get("latency_ms", 0) for r in results), 1
                )
                gain = expert_sum_ms / wall_ms if wall_ms > 0 else 1.0

                # Log
                matched = [r for r in results if r.get("match")]
                if matched:
                    skills_str = ", ".join(
                        f"{r['skill']}.{r.get('action', '?')}" for r in matched
                    )
                    print(f"  [{wall_ms}ms wall, {expert_sum_ms}ms sum, "
                          f"{gain:.1f}× gain] \"{transcript[:50]}\" -> {skills_str}")
                else:
                    print(f"  [{wall_ms}ms wall] \"{transcript[:50]}\" -> (none)")

                self._send_json(200, {
                    "results": results,
                    "wall_ms": wall_ms,
                    "expert_sum_ms": expert_sum_ms,
                    "parallel_gain": round(gain, 2),
                })

            except Exception as e:
                self._send_json(500, {"error": str(e)})

        def _send_json(self, status: int, data: dict) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, format: str, *args: Any) -> None:
            pass  # suppress default HTTP logging

    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"Expert server running on http://localhost:{port}")
        print(f"  Workers: {len(active_skills)} (parallel inference)")
        print(f"  Skills: {', '.join(active_skills)}")
        print(f"  Endpoints:")
        for skill_name in active_skills:
            print(f"    POST /v1/classify/{skill_name}  — {skill_name} expert only")
        print(f"    POST /v1/classify           — all skills in parallel")
        print(f"    GET  /health                — server status")
        print()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Shutting down workers...")
            pool.shutdown()
            print("  Expert server stopped.")


def main() -> None:
    """CLI entry point."""
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Serve skill expert models")
    parser.add_argument("--port", type=int, default=8234)
    parser.add_argument("--skills", help="Comma-separated skill names")
    args = parser.parse_args()

    skills = args.skills.split(",") if args.skills else None
    run_server(port=args.port, skills=skills)


if __name__ == "__main__":
    main()
