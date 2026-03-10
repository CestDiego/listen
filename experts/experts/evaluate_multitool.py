"""
Multi-tool model evaluation — parse <tool_call> blocks, compare to expected.

Metrics:
  - Exact match accuracy (predicted tool set == expected tool set)
  - Per-tool precision, recall, F1
  - Dual-activation accuracy
  - False positive / false negative analysis
  - Average latency per inference

Single metric: exact-match accuracy (all expected tools called, no extras).

Usage:
  uv run evaluate-multitool
  uv run evaluate-multitool --adapter models/multitool/adapters.safetensors
  uv run evaluate-multitool --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import BASE_MODEL, DATA_DIR, MODELS_DIR, TOOL_DEFINITIONS


# ── Tool call parsing ─────────────────────────────────────────────

# Allowlist of valid tool names — rejects hallucinated actions like "accommodator.unknown"
VALID_TOOLS = {t["function"]["name"] for t in TOOL_DEFINITIONS}

# Regex: only match word chars + dots in function name (no injection)
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([\w.]+)>\s*(?:<parameter=[^>]*>[^<]*</parameter>\s*)*</function>\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(raw: str, filter_valid: bool = True) -> list[str]:
    """
    Extract tool call function names from model output.

    Handles:
      - <think>...</think> prefix (Qwen3.5 always emits)
      - Multiple <tool_call> blocks
      - "No tools needed." → empty list
      - Filters to valid tool names only (prevents hallucinated actions)
    """
    text = raw.strip()

    # Strip ALL thinking blocks (not just the first)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Check for explicit no-tool response
    if "no tools needed" in text.lower():
        return []

    # Extract all tool calls
    matches = TOOL_CALL_RE.findall(text)

    # Filter to valid tools only — prevents hallucinated actions
    if filter_valid:
        valid = [m for m in matches if m in VALID_TOOLS]
        rejected = [m for m in matches if m not in VALID_TOOLS]
        if rejected:
            import sys
            print(f"  [warn] rejected hallucinated tools: {rejected}", file=sys.stderr)
        return valid

    return list(matches)


# ── Model loading ─────────────────────────────────────────────────

def load_model_and_tokenizer(adapter_path: str | None = None):
    """Load the base model with optional LoRA adapters."""
    from mlx_lm import load

    if adapter_path and Path(adapter_path).exists():
        model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)
    else:
        model, tokenizer = load(BASE_MODEL)

    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────

def run_inference(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    transcript: str,
    max_tokens: int = 120,
) -> tuple[str, float]:
    """Run a single inference and return (output_text, latency_seconds)."""
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    def greedy_sampler(logits: Any) -> Any:
        return mx.argmax(logits, axis=-1)

    start = time.perf_counter()
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=greedy_sampler,
        verbose=False,
    )
    elapsed = time.perf_counter() - start

    return output, elapsed


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_multitool(
    adapter_path: str | None = None,
    verbose: bool = False,
    split: str = "test",
) -> dict[str, Any]:
    """
    Evaluate the unified multi-tool model.

    Returns dict with: exact_match_accuracy, per_tool metrics, latency, etc.
    """
    from .generate_multitool import SYSTEM_PROMPT

    # Load data
    data_path = DATA_DIR / "multitool" / f"{split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"No {split} data at {data_path}. Run: uv run generate-multitool"
        )

    entries = []
    with open(data_path) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Loading model (adapter: {adapter_path or 'none'})...")
    model, tokenizer = load_model_and_tokenizer(adapter_path)

    # Metrics
    exact_matches = 0
    total = len(entries)
    latencies: list[float] = []

    # Per-tool confusion
    all_tools = set()
    tool_tp: dict[str, int] = {}
    tool_fp: dict[str, int] = {}
    tool_fn: dict[str, int] = {}

    # Category tracking
    n_negative = 0
    n_negative_correct = 0
    n_single = 0
    n_single_correct = 0
    n_dual = 0
    n_dual_correct = 0

    errors: list[dict] = []

    print(f"Running {total} eval cases...")
    for i, entry in enumerate(entries):
        messages = entry["messages"]
        transcript = messages[1]["content"]
        expected_content = messages[2]["content"]

        # Parse expected tool calls
        expected_tools = set(parse_tool_calls(expected_content))
        if not expected_tools and "no tools needed" in expected_content.lower():
            expected_tools = set()

        # Run inference
        output_text, latency = run_inference(model, tokenizer, SYSTEM_PROMPT, transcript)
        latencies.append(latency)

        # Parse predicted tool calls
        predicted_tools = set(parse_tool_calls(output_text))

        # Exact match
        is_exact = predicted_tools == expected_tools
        if is_exact:
            exact_matches += 1

        # Category tracking
        n_expected = len(expected_tools)
        if n_expected == 0:
            n_negative += 1
            if is_exact:
                n_negative_correct += 1
        elif n_expected == 1:
            n_single += 1
            if is_exact:
                n_single_correct += 1
        else:
            n_dual += 1
            if is_exact:
                n_dual_correct += 1

        # Per-tool metrics
        for tool in expected_tools | predicted_tools:
            all_tools.add(tool)
            if tool not in tool_tp:
                tool_tp[tool] = tool_fp[tool] = tool_fn[tool] = 0

            if tool in expected_tools and tool in predicted_tools:
                tool_tp[tool] += 1
            elif tool in predicted_tools and tool not in expected_tools:
                tool_fp[tool] += 1
            elif tool in expected_tools and tool not in predicted_tools:
                tool_fn[tool] += 1

        # Error tracking
        if not is_exact:
            errors.append({
                "transcript": transcript[:80],
                "expected": sorted(expected_tools),
                "predicted": sorted(predicted_tools),
                "raw": output_text[:200],
            })

        if verbose:
            status = "PASS" if is_exact else "FAIL"
            print(f"  [{i+1}/{total}] {status} | expected={sorted(expected_tools)} predicted={sorted(predicted_tools)} | {transcript[:60]}")

        if not verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}]...")

    # Compute metrics
    exact_match_acc = exact_matches / max(1, total)
    avg_latency = sum(latencies) / max(1, len(latencies))

    # Per-tool F1
    per_tool: dict[str, dict[str, float]] = {}
    for tool in sorted(all_tools):
        tp = tool_tp.get(tool, 0)
        fp = tool_fp.get(tool, 0)
        fn = tool_fn.get(tool, 0)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        per_tool[tool] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    results = {
        "adapter": adapter_path,
        "split": split,
        "total": total,
        "exact_matches": exact_matches,
        "exact_match_accuracy": round(exact_match_acc, 4),
        "negative": {"total": n_negative, "correct": n_negative_correct},
        "single_tool": {"total": n_single, "correct": n_single_correct},
        "dual_tool": {"total": n_dual, "correct": n_dual_correct},
        "per_tool": per_tool,
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "errors": errors[:10],  # Cap for readability
    }

    return results


def print_results(results: dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"MULTI-TOOL EVAL RESULTS ({results['split']} split)")
    print(f"{'='*60}")
    print(f"  Adapter:              {results['adapter'] or 'baseline (no adapter)'}")
    print(f"  Total cases:          {results['total']}")
    print(f"  Exact match accuracy: {results['exact_match_accuracy']:.1%} ({results['exact_matches']}/{results['total']})")
    print()

    neg = results["negative"]
    single = results["single_tool"]
    dual = results["dual_tool"]
    print(f"  Negatives:     {neg['correct']}/{neg['total']} correct")
    print(f"  Single-tool:   {single['correct']}/{single['total']} correct")
    print(f"  Dual-tool:     {dual['correct']}/{dual['total']} correct")
    print()

    print(f"  Per-tool metrics:")
    for tool, m in results["per_tool"].items():
        print(f"    {tool:25s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    print()

    print(f"  Avg latency: {results['avg_latency_ms']:.0f}ms")

    if results["errors"]:
        print(f"\n  Errors ({len(results['errors'])} shown):")
        for err in results["errors"]:
            print(f"    - \"{err['transcript']}\"")
            print(f"      expected: {err['expected']}")
            print(f"      predicted: {err['predicted']}")

    print(f"{'='*60}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate multi-tool model")
    parser.add_argument("--adapter", help="Path to LoRA adapter")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = evaluate_multitool(
        adapter_path=args.adapter,
        verbose=args.verbose,
        split=args.split,
    )
    print_results(results)

    # Save results
    out_path = MODELS_DIR / "multitool" / f"eval_{results['split']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
