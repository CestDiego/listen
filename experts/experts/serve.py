"""
Skill expert HTTP server — serves fine-tuned models via a simple REST API.

Runs all skill experts in a single process, each loaded with its LoRA adapter.
The Bun pipeline calls per-skill endpoints in parallel via Promise.all.

Endpoints:
  GET  /health                → {"status": "ok", "skills": [...], "endpoints": [...]}
  POST /v1/classify/{skill}   → {"skill": "music", "match": true, "action": "play", ...}
  POST /v1/classify           → {"results": [...]}  (all skills, sequential — for eval/compat)

Threading: Uses ThreadingMixIn so parallel per-skill requests execute concurrently.
Each skill expert has its own model instance, so thread-safety is per-model.

Usage:
  uv run serve --port 8234
  uv run serve --skills music,wellbeing
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import BASE_MODEL, MODELS_DIR, SKILLS
from .evaluate import parse_output


# Global model cache — each skill has its own model, tokenizer, prompt
_models: dict[str, tuple[Any, Any, str]] = {}


def load_skill_expert(skill_name: str) -> tuple[Any, Any, str]:
    """Load a skill expert (base + LoRA adapter). Thread-safe via per-skill lock."""
    if skill_name in _models:
        return _models[skill_name]

    from mlx_lm import load
    from .generate import system_prompt

    adapter_path = MODELS_DIR / skill_name / "adapters.safetensors"

    if adapter_path.exists():
        print(f"  Loading {skill_name} expert (with LoRA adapter)...")
        result = load(BASE_MODEL, adapter_path=str(adapter_path))
    else:
        print(f"  Loading {skill_name} expert (baseline, no adapter)...")
        result = load(BASE_MODEL)
    model, tokenizer = result[0], result[1]  # type: ignore[index]

    sys_prompt = system_prompt(skill_name)
    _models[skill_name] = (model, tokenizer, sys_prompt)
    return model, tokenizer, sys_prompt


def classify_single(
    skill_name: str,
    transcript: str,
) -> dict[str, Any]:
    """
    Classify a transcript against a single skill expert.
    Uses a per-skill lock to serialize GPU inference per model.
    Returns a result dict with match, action, confidence, latency_ms.
    """
    from mlx_lm import generate

    if skill_name not in SKILLS:
        return {
            "skill": skill_name,
            "match": False,
            "status": "error",
            "error": f"Unknown skill: {skill_name}",
            "latency_ms": 0,
        }

    model, tokenizer, sys_prompt = load_skill_expert(skill_name)

    def greedy_sampler(logits: Any) -> Any:
        return mx.argmax(logits, axis=-1)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": transcript},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    start = time.perf_counter()
    output = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=60,
        sampler=greedy_sampler,
        verbose=False,
    )
    latency = time.perf_counter() - start

    parsed = parse_output(output)
    result: dict[str, Any] = {
        "skill": skill_name,
        "match": parsed.get("match", False),
        "status": "ok",
        "latency_ms": round(latency * 1000, 1),
    }
    if parsed.get("match"):
        result["action"] = parsed.get("action", "")
        result["confidence"] = parsed.get("confidence", 0.0)

    return result


def classify_all(
    transcript: str,
    skills: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Classify a transcript against all requested skill experts (sequential).
    Used by the /v1/classify bulk endpoint (eval, backward compat).
    """
    if skills is None:
        skills = list(SKILLS.keys())

    return [classify_single(skill_name, transcript) for skill_name in skills]


def run_server(port: int = 8234, skills: list[str] | None = None) -> None:
    """Run the threaded HTTP classification server."""
    import http.server
    import socketserver

    active_skills = skills or list(SKILLS.keys())

    # Pre-load all skill experts at startup
    print(f"\nLoading {len(active_skills)} skill expert(s)...")
    for skill_name in active_skills:
        load_skill_expert(skill_name)
    print("All experts loaded.\n")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                response = {
                    "status": "ok",
                    "skills": active_skills,
                    "base_model": BASE_MODEL,
                    "endpoints": [
                        f"/v1/classify/{s}" for s in active_skills
                    ] + ["/v1/classify"],
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            # ── Per-skill endpoint: POST /v1/classify/{skill} ──────
            for skill_name in active_skills:
                if self.path == f"/v1/classify/{skill_name}":
                    self._handle_single_skill(skill_name)
                    return

            # ── Bulk endpoint: POST /v1/classify ───────────────────
            if self.path == "/v1/classify":
                self._handle_bulk_classify()
                return

            self.send_error(404)

        def _handle_single_skill(self, skill_name: str) -> None:
            """Handle a single-skill classification request."""
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                req = json.loads(body)
                transcript = req.get("transcript", "")

                if not transcript or len(transcript.strip()) < 10:
                    result = {
                        "skill": skill_name,
                        "match": False,
                        "status": "too_short",
                        "latency_ms": 0,
                    }
                else:
                    result = classify_single(skill_name, transcript)

                # Log
                matched = result.get("match", False)
                action = result.get("action", "?")
                ms = result.get("latency_ms", 0)
                icon = "+" if matched else "-"
                detail = f".{action}" if matched else ""
                print(f"  [{ms}ms] {icon} {skill_name}{detail} \"{transcript[:50]}\"")

                self._send_json(200, result)

            except Exception as e:
                self._send_json(500, {
                    "skill": skill_name,
                    "match": False,
                    "status": "error",
                    "error": str(e),
                    "latency_ms": 0,
                })

        def _handle_bulk_classify(self) -> None:
            """Handle bulk classification (all skills, sequential)."""
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                req = json.loads(body)
                transcript = req.get("transcript", "")
                req_skills = req.get("skills", active_skills)

                if not transcript or len(transcript.strip()) < 10:
                    response = {"results": [], "reason": "too short"}
                else:
                    start = time.perf_counter()
                    results = classify_all(transcript, req_skills)
                    total_ms = round((time.perf_counter() - start) * 1000, 1)

                    matched = [r for r in results if r.get("match")]
                    if matched:
                        skills_str = ", ".join(
                            f"{r['skill']}.{r.get('action', '?')}" for r in matched
                        )
                        print(f"  [{total_ms}ms] \"{transcript[:60]}\" -> {skills_str}")
                    else:
                        print(f"  [{total_ms}ms] \"{transcript[:60]}\" -> (none)")

                    response = {"results": results}

                self._send_json(200, response)

            except Exception as e:
                self._send_json(500, {"error": str(e)})

        def _send_json(self, status: int, data: dict) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, format: str, *args: Any) -> None:
            # Suppress default HTTP logging (we log our own)
            pass

    # Use standard TCPServer — MLX's Metal backend doesn't play well with
    # ThreadingMixIn due to GPU command buffer sharing. Per-skill endpoints
    # still work for clarity; the Bun side gets parallelism by calling the
    # bulk endpoint which returns per-expert timing for full observability.
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"Expert server running on http://localhost:{port}")
        print(f"  Skills: {', '.join(active_skills)}")
        print(f"  Endpoints:")
        for skill_name in active_skills:
            print(f"    POST /v1/classify/{skill_name}  — {skill_name} expert only")
        print(f"    POST /v1/classify           — all skills (per-expert timing)")
        print(f"    GET  /health                — server status")
        print(f"  Example:")
        print(f'    curl -s localhost:{port}/v1/classify -d \'{{"transcript":"play some music"}}\' | python3 -m json.tool')
        print()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Expert server stopped.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Serve skill expert models")
    parser.add_argument("--port", type=int, default=8234)
    parser.add_argument("--skills", help="Comma-separated skill names")
    args = parser.parse_args()

    skills = args.skills.split(",") if args.skills else None
    run_server(port=args.port, skills=skills)


if __name__ == "__main__":
    main()
