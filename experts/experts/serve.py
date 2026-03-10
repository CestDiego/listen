"""
Skill expert HTTP server — serves fine-tuned models via a simple REST API.

Runs all skill experts in a single process, each loaded with its LoRA adapter.
The Bun pipeline calls this instead of LM Studio for routing.

Endpoints:
  GET  /health       → {"status": "ok", "skills": ["music", "wellbeing"]}
  POST /v1/classify  → {"results": [{"skill": "music", "match": true, "action": "play", "confidence": 0.95}]}

Each skill expert runs independently — no shared state.

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


# Global model cache
_models: dict[str, tuple[Any, Any, str]] = {}  # skill -> (model, tokenizer, sys_prompt)


def load_skill_expert(skill_name: str) -> tuple[Any, Any, str]:
    """Load a skill expert (base + LoRA adapter)."""
    if skill_name in _models:
        return _models[skill_name]

    from mlx_lm import load
    from .generate import system_prompt

    adapter_path = MODELS_DIR / skill_name / "adapters.safetensors"

    if adapter_path.exists():
        print(f"  Loading {skill_name} expert (with LoRA adapter)...")
        model, tokenizer = load(BASE_MODEL, adapter_path=str(adapter_path))
    else:
        print(f"  Loading {skill_name} expert (baseline, no adapter)...")
        model, tokenizer = load(BASE_MODEL)

    sys_prompt = system_prompt(skill_name)
    _models[skill_name] = (model, tokenizer, sys_prompt)
    return model, tokenizer, sys_prompt


def classify_transcript(
    transcript: str,
    skills: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Classify a transcript against all requested skill experts.

    Returns a list of results, one per skill.
    """
    from mlx_lm import generate

    if skills is None:
        skills = list(SKILLS.keys())

    def greedy_sampler(logits: Any) -> Any:
        return mx.argmax(logits, axis=-1)

    results = []
    for skill_name in skills:
        if skill_name not in SKILLS:
            results.append({
                "skill": skill_name,
                "match": False,
                "error": f"Unknown skill: {skill_name}",
            })
            continue

        model, tokenizer, sys_prompt = load_skill_expert(skill_name)

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
            "latency_ms": round(latency * 1000, 1),
        }
        if parsed.get("match"):
            result["action"] = parsed.get("action", "")
            result["confidence"] = parsed.get("confidence", 0.0)

        results.append(result)

    return results


def run_server(port: int = 8234, skills: list[str] | None = None) -> None:
    """Run the HTTP classification server."""
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
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/v1/classify":
                self.send_error(404)
                return

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
                    results = classify_transcript(transcript, req_skills)
                    total_ms = round((time.perf_counter() - start) * 1000, 1)

                    # Log to stdout
                    matched = [r for r in results if r.get("match")]
                    if matched:
                        skills_str = ", ".join(
                            f"{r['skill']}.{r.get('action', '?')}" for r in matched
                        )
                        print(f"  [{total_ms}ms] \"{transcript[:60]}\" -> {skills_str}")
                    else:
                        print(f"  [{total_ms}ms] \"{transcript[:60]}\" -> (none)")

                    response = {"results": results}

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        def log_message(self, format: str, *args: Any) -> None:
            # Suppress default HTTP logging (we log our own)
            pass

    # Allow port reuse for fast restarts
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"Expert server running on http://localhost:{port}")
        print(f"  Skills: {', '.join(active_skills)}")
        print(f"  Endpoints:")
        print(f"    GET  /health      — server status")
        print(f"    POST /v1/classify — classify transcript")
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
