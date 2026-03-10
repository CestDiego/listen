"""
Skill expert evaluation — READ-ONLY per autoresearch convention.

Loads a skill expert model (base + optional LoRA adapter) and runs it
against the test set. Reports:
  - Accuracy, precision, recall, F1
  - Per-action accuracy (for multi-action skills)
  - False positive analysis
  - Average latency per inference

Single metric: F1 score (balances precision and recall).

Usage:
  uv run evaluate --skill music
  uv run evaluate --skill music --adapter models/music/adapters.safetensors
  uv run evaluate --skill wellbeing --verbose
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from .config import BASE_MODEL, DATA_DIR, MODELS_DIR


def load_model_and_tokenizer(
    skill_name: str,
    adapter_path: str | None = None,
):
    """Load the base model with optional LoRA adapters."""
    from mlx_lm import load

    model_path = adapter_path or BASE_MODEL

    if adapter_path and Path(adapter_path).exists():
        # Load base model + LoRA adapter
        model, tokenizer = load(
            BASE_MODEL,
            adapter_path=adapter_path,
        )
    else:
        # Baseline: just the base model
        model, tokenizer = load(BASE_MODEL)

    return model, tokenizer


def run_inference(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    transcript: str,
    max_tokens: int = 60,
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

    # Use greedy sampling (argmax) for deterministic eval
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


def parse_output(raw: str) -> dict[str, Any]:
    """Parse the model's JSON output, handling thinking tokens."""
    text = raw.strip()

    # Strip thinking tokens if present
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()

    # Find first JSON object
    start = text.find("{")
    if start == -1:
        return {"match": False, "_parse_error": True}

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return {"match": False, "_parse_error": True}

    return {"match": False, "_parse_error": True}


def evaluate_skill(
    skill_name: str,
    adapter_path: str | None = None,
    verbose: bool = False,
    split: str = "test",
) -> dict[str, Any]:
    """
    Evaluate a skill expert and return metrics.

    Returns dict with: accuracy, precision, recall, f1, avg_latency,
                       per_action_accuracy, false_positives, false_negatives
    """
    from .generate import system_prompt

    # Load data
    data_path = DATA_DIR / skill_name / f"{split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"No {split} data at {data_path}. Run: uv run generate")

    entries = []
    with open(data_path) as f:
        for line in f:
            entries.append(json.loads(line))

    print(f"Loading model for '{skill_name}' (adapter: {adapter_path or 'none'})...")
    model, tokenizer = load_model_and_tokenizer(skill_name, adapter_path)

    sys_prompt = system_prompt(skill_name)
    tp, fp, tn, fn = 0, 0, 0, 0
    action_correct = 0
    action_total = 0
    parse_errors = 0
    latencies: list[float] = []
    false_positives: list[dict] = []
    false_negatives: list[dict] = []

    print(f"Running {len(entries)} eval cases...")
    for i, entry in enumerate(entries):
        messages = entry["messages"]
        transcript = messages[1]["content"]
        expected = json.loads(messages[2]["content"])
        expected_match = expected.get("match", False)
        expected_action = expected.get("action", "")

        output_text, latency = run_inference(model, tokenizer, sys_prompt, transcript)
        latencies.append(latency)

        predicted = parse_output(output_text)
        predicted_match = predicted.get("match", False)
        predicted_action = predicted.get("action", "")

        if predicted.get("_parse_error"):
            parse_errors += 1

        # Confusion matrix
        if expected_match and predicted_match:
            tp += 1
            if expected_action and predicted_action == expected_action:
                action_correct += 1
            if expected_action:
                action_total += 1
        elif not expected_match and not predicted_match:
            tn += 1
        elif not expected_match and predicted_match:
            fp += 1
            false_positives.append({
                "transcript": transcript,
                "predicted_action": predicted_action,
                "raw": output_text[:200],
            })
        else:  # expected_match and not predicted_match
            fn += 1
            false_negatives.append({
                "transcript": transcript,
                "expected_action": expected_action,
                "raw": output_text[:200],
            })

        if verbose:
            status = "PASS" if (expected_match == predicted_match) else "FAIL"
            print(f"  [{i+1}/{len(entries)}] {status} | expected={expected_match} predicted={predicted_match} | {transcript[:60]}...")

        # Periodic progress
        if not verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(entries)}]...")

    # Compute metrics
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    action_acc = action_correct / max(1, action_total)
    avg_latency = sum(latencies) / max(1, len(latencies))

    results = {
        "skill": skill_name,
        "adapter": adapter_path,
        "split": split,
        "total": len(entries),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "action_accuracy": round(action_acc, 4),
        "parse_errors": parse_errors,
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "false_positives": false_positives[:5],  # Cap for readability
        "false_negatives": false_negatives[:5],
    }

    return results


def print_results(results: dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"EVAL RESULTS: {results['skill']} ({results['split']} split)")
    print(f"{'='*60}")
    print(f"  Adapter:        {results['adapter'] or 'baseline (no adapter)'}")
    print(f"  Total cases:    {results['total']}")
    print(f"  TP: {results['tp']}  FP: {results['fp']}  TN: {results['tn']}  FN: {results['fn']}")
    print(f"  Accuracy:       {results['accuracy']:.1%}")
    print(f"  Precision:      {results['precision']:.1%}")
    print(f"  Recall:         {results['recall']:.1%}")
    print(f"  F1:             {results['f1']:.1%}")
    print(f"  Action acc:     {results['action_accuracy']:.1%}")
    print(f"  Parse errors:   {results['parse_errors']}")
    print(f"  Avg latency:    {results['avg_latency_ms']:.0f}ms")

    if results["false_positives"]:
        print(f"\n  False Positives ({len(results['false_positives'])} shown):")
        for fp_case in results["false_positives"]:
            print(f"    - \"{fp_case['transcript'][:70]}...\" -> {fp_case['predicted_action']}")

    if results["false_negatives"]:
        print(f"\n  False Negatives ({len(results['false_negatives'])} shown):")
        for fn_case in results["false_negatives"]:
            print(f"    - \"{fn_case['transcript'][:70]}...\" (expected: {fn_case['expected_action']})")

    print(f"{'='*60}")


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a skill expert model")
    parser.add_argument("--skill", required=True, choices=["music", "wellbeing"])
    parser.add_argument("--adapter", help="Path to LoRA adapter file")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = evaluate_skill(
        skill_name=args.skill,
        adapter_path=args.adapter,
        verbose=args.verbose,
        split=args.split,
    )
    print_results(results)

    # Save results
    out_path = MODELS_DIR / args.skill / f"eval_{results['split']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
