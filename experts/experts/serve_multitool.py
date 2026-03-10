"""
Multi-tool HTTP server — single model, no worker pool.

Architecture:
  - One process, one model loaded at startup
  - Single inference per request → outputs 0..N tool calls
  - Much simpler than the parallel worker pool (no multiprocessing)

Endpoints:
  GET  /health         → {"status": "ok", "model": "multitool", ...}
  POST /v1/classify    → multi-tool classification (backward-compatible response)

The response format is backward-compatible with the per-expert server:
  {
    "results": [
      {"skill": "music", "match": true, "action": "skip", "latency_ms": 150},
      {"skill": "wellbeing", "match": true, "action": "check_in", "latency_ms": 150}
    ],
    "wall_ms": 150,
    "expert_sum_ms": 150,
    "parallel_gain": 1.0
  }

Usage:
  uv run python -m experts.serve_multitool --port 8234
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import mlx.core as mx

from .config import BASE_MODEL, MODELS_DIR, SKILLS
from .evaluate_multitool import parse_tool_calls


# ── Model singleton ───────────────────────────────────────────────

_model = None
_tokenizer = None
_system_prompt = None


def load_model(adapter_path: str | None = None) -> None:
    """Load the multi-tool model at startup."""
    global _model, _tokenizer, _system_prompt
    from mlx_lm import load
    from .generate_multitool import SYSTEM_PROMPT

    actual_adapter = adapter_path
    if not actual_adapter:
        default = MODELS_DIR / "multitool" / "adapters.safetensors"
        if default.exists():
            actual_adapter = str(default)

    if actual_adapter:
        print(f"  Loading {BASE_MODEL} + adapter {actual_adapter}")
        _model, _tokenizer = load(BASE_MODEL, adapter_path=actual_adapter)
    else:
        print(f"  Loading {BASE_MODEL} (no adapter)")
        _model, _tokenizer = load(BASE_MODEL)

    _system_prompt = SYSTEM_PROMPT

    # Warm up with a dummy inference
    print("  Warming up...")
    _classify("warmup test")
    print("  Ready.")


def _classify(transcript: str) -> tuple[list[str], float]:
    """Run inference and return (tool_names, latency_ms)."""
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": _system_prompt},
        {"role": "user", "content": transcript},
    ]
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    def greedy_sampler(logits: Any) -> Any:
        return mx.argmax(logits, axis=-1)

    start = time.perf_counter()
    output = generate(
        _model,
        _tokenizer,
        prompt=prompt,
        max_tokens=120,
        sampler=greedy_sampler,
        verbose=False,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    tool_calls = parse_tool_calls(output)
    return tool_calls, latency_ms


def _tool_call_to_result(tool_name: str, latency_ms: float) -> dict[str, Any]:
    """Convert a tool call name like 'music.skip' to a backward-compatible result."""
    parts = tool_name.split(".", 1)
    if len(parts) == 2:
        skill, action = parts
    else:
        skill, action = tool_name, ""

    return {
        "skill": skill,
        "match": True,
        "action": action,
        "confidence": 0.95,
        "latency_ms": latency_ms,
        "status": "ok",
    }


# ── HTTP server ───────────────────────────────────────────────────

def run_server(port: int = 8234, adapter_path: str | None = None) -> None:
    """Run the multi-tool HTTP server."""
    import http.server
    import socketserver

    load_model(adapter_path)

    all_skills = list(SKILLS.keys())

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                response = {
                    "status": "ok",
                    "model": "multitool",
                    "base_model": BASE_MODEL,
                    "skills": all_skills,
                    "parallel": False,
                    "workers": 1,
                    "endpoints": ["/v1/classify"],
                }
                self._send_json(200, response)
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path == "/v1/classify":
                self._handle_classify()
            elif self.path.startswith("/v1/classify/"):
                # Backward compat: per-skill endpoint → just run full classify
                self._handle_classify()
            else:
                self.send_error(404)

        def _handle_classify(self) -> None:
            MAX_BODY_SIZE = 64 * 1024  # 64KB — more than enough for any transcript
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > MAX_BODY_SIZE:
                self._send_json(413, {"error": "request too large"})
                return
            body = self.rfile.read(content_length)

            try:
                req = json.loads(body)
                transcript = req.get("transcript", "")

                if not transcript or len(transcript.strip()) < 10:
                    self._send_json(200, {"results": [], "reason": "too short"})
                    return

                tool_calls, latency_ms = _classify(transcript)

                # Build backward-compatible results
                results = [
                    _tool_call_to_result(tc, latency_ms)
                    for tc in tool_calls
                ]

                # Also include non-matching skills for full observability
                matched_skills = {tc.split(".")[0] for tc in tool_calls}
                for skill in all_skills:
                    if skill not in matched_skills:
                        results.append({
                            "skill": skill,
                            "match": False,
                            "latency_ms": latency_ms,
                            "status": "ok",
                        })

                # Log
                if tool_calls:
                    tools_str = ", ".join(tool_calls)
                    print(f"  [{latency_ms}ms] \"{transcript[:50]}\" -> {tools_str}")
                else:
                    print(f"  [{latency_ms}ms] \"{transcript[:50]}\" -> (none)")

                self._send_json(200, {
                    "results": results,
                    "wall_ms": latency_ms,
                    "expert_sum_ms": latency_ms,
                    "parallel_gain": 1.0,
                })

            except Exception as e:
                import traceback
                traceback.print_exc()  # log full error internally
                self._send_json(500, {"error": "internal server error"})

        def _send_json(self, status: int, data: dict) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, format: str, *args: Any) -> None:
            pass  # suppress default HTTP logging

    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"\nMulti-tool server running on http://localhost:{port}")
        print(f"  Model: single unified multi-tool")
        print(f"  Skills: {', '.join(all_skills)}")
        print(f"  Endpoints:")
        print(f"    POST /v1/classify  — multi-tool classification")
        print(f"    GET  /health       — server status")
        print()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Multi-tool server stopped.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Serve multi-tool model")
    parser.add_argument("--port", type=int, default=8234)
    parser.add_argument("--adapter", help="Path to LoRA adapter")
    args = parser.parse_args()

    run_server(port=args.port, adapter_path=args.adapter)


if __name__ == "__main__":
    main()
