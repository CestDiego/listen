"""
Autoresearch experiment loop — AGENT-EDITABLE.

This is the agent's main entry point. It follows Karpathy's autoresearch
philosophy: train for a fixed time budget, eval against an immutable metric,
keep or discard, log everything, never stop.

The agent modifies ONLY `train_multitool.py` (hyperparams, LoRA config).
This file orchestrates the loop and is also agent-editable (the agent can
improve the loop itself, e.g. add smarter analysis).

Fixed constraints (DO NOT CHANGE):
  - Evaluation function lives in evaluate_multitool.py (read-only)
  - Eval cases live in eval-cases.json (read-only)
  - Data generation lives in generate_multitool.py (read-only)
  - The single metric is exact-match accuracy

Usage:
  uv run experiment                       # one cycle: train → eval → keep/discard
  uv run experiment --iters 100           # more training iters
  uv run experiment --eval-only           # just eval current adapter
  uv run experiment --baseline            # eval base model, cache result
  uv run experiment --loop 10             # run 10 experiments autonomously
  uv run experiment --note "try rank 32"  # attach a note to the results log
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import BASE_MODEL, DATA_DIR, MODELS_DIR, TOOL_DEFINITIONS
from .evaluate_multitool import (
    load_model_and_tokenizer,
    run_inference,
    parse_tool_calls,
)
from .generate_multitool import SYSTEM_PROMPT


# ── Paths ─────────────────────────────────────────────────────────

MULTITOOL_DIR = MODELS_DIR / "multitool"
ADAPTER_PATH = str(MULTITOOL_DIR / "adapters.safetensors")
BEST_ADAPTER_PATH = MULTITOOL_DIR / "best_adapters.safetensors"
BASELINE_CACHE = MULTITOOL_DIR / "baseline_eval.json"
RESULTS_TSV = MULTITOOL_DIR / "results.tsv"
EVAL_CASES_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "listen" / "skills" / "eval-cases.json"
)


# ── Results ledger (append-only) ──────────────────────────────────

TSV_COLUMNS = [
    "timestamp", "run_id", "iters", "lr", "lora_rank", "lora_layers",
    "accuracy", "exact", "total", "elapsed_s", "status", "note",
]


def _init_results_tsv() -> None:
    """Create results.tsv with header if it doesn't exist."""
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_TSV, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(TSV_COLUMNS)


def _append_result(row: dict[str, Any]) -> None:
    """Append one row to results.tsv."""
    _init_results_tsv()
    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([row.get(c, "") for c in TSV_COLUMNS])


def _read_best_accuracy() -> float:
    """Read the best accuracy from results.tsv (kept runs only)."""
    if not RESULTS_TSV.exists():
        return 0.0
    best = 0.0
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("status") == "keep":
                try:
                    best = max(best, float(row["accuracy"]))
                except (ValueError, KeyError):
                    pass
    return best


# ── Baseline caching ──────────────────────────────────────────────

def _load_baseline() -> float | None:
    """Load cached baseline accuracy, or None if not cached."""
    if not BASELINE_CACHE.exists():
        return None
    data = json.loads(BASELINE_CACHE.read_text())
    return data.get("accuracy")


def _save_baseline(accuracy: float, results: dict) -> None:
    """Cache baseline evaluation results."""
    BASELINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_CACHE.write_text(json.dumps({
        "accuracy": accuracy,
        "exact": results["exact"],
        "total": results["total"],
        "timestamp": datetime.now().isoformat(),
        "tool_tp": results["tool_tp"],
        "tool_fp": results["tool_fp"],
        "tool_fn": results["tool_fn"],
    }, indent=2))


# ── Best adapter management (keep/discard) ────────────────────────

def _snapshot_best(accuracy: float) -> None:
    """Copy current adapter as the best known adapter."""
    src = Path(ADAPTER_PATH)
    if not src.exists():
        return
    BEST_ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        # Adapter might be a directory with checkpoints
        if BEST_ADAPTER_PATH.exists():
            shutil.rmtree(BEST_ADAPTER_PATH)
        shutil.copytree(src, BEST_ADAPTER_PATH)
    else:
        shutil.copy2(src, BEST_ADAPTER_PATH)

    # Write metadata alongside
    meta = BEST_ADAPTER_PATH.parent / "best_meta.json"
    meta.write_text(json.dumps({
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))


# ── Training ──────────────────────────────────────────────────────

def quick_train(iters: int = 50, lr: float = 5e-5, resume: bool = False) -> float:
    """Train for a small number of iters. Returns elapsed seconds."""
    from .train_multitool import train_multitool
    start = time.perf_counter()
    train_multitool(iters=iters, lr=lr, resume=resume)
    return time.perf_counter() - start


# ── Evaluation (uses immutable eval from evaluate_multitool.py) ───

def load_eval_cases() -> list[dict[str, Any]]:
    """Load the curated eval cases from eval-cases.json.

    These are the ground-truth, hand-curated cases — the IMMUTABLE test set.
    DO NOT use the synthetic test.jsonl split here; that may contain
    labeling errors from augmentation.
    """
    with open(EVAL_CASES_PATH) as f:
        data = json.load(f)

    cases = []
    for c in data["cases"]:
        if "_section" in c:
            continue
        transcript = c["transcript"]
        skills = c["expect"].get("skills", [])
        expected_tools = set()
        for s in skills:
            expected_tools.add(f"{s['skill']}.{s['action']}")
        cases.append({
            "name": c["name"],
            "transcript": transcript,
            "expected": expected_tools,
        })
    return cases


def eval_adapter(adapter_path: str | None, verbose: bool = False) -> dict[str, Any]:
    """
    Eval on the curated eval cases.
    Returns structured results with the single metric: exact-match accuracy.
    """
    cases = load_eval_cases()

    print(f"\n  Loading model (adapter: {adapter_path or 'none'})...")
    model, tokenizer = load_model_and_tokenizer(adapter_path)

    exact = 0
    total = len(cases)
    latencies: list[float] = []

    tool_tp: Counter = Counter()
    tool_fp: Counter = Counter()
    tool_fn: Counter = Counter()

    errors: list[dict] = []

    print(f"  Running {total} eval cases...\n")
    for i, case in enumerate(cases):
        output, latency = run_inference(
            model, tokenizer, SYSTEM_PROMPT, case["transcript"]
        )
        latencies.append(latency)
        predicted = set(parse_tool_calls(output))
        expected = case["expected"]

        is_match = predicted == expected
        if is_match:
            exact += 1

        for tool in expected | predicted:
            if tool in expected and tool in predicted:
                tool_tp[tool] += 1
            elif tool in predicted:
                tool_fp[tool] += 1
            else:
                tool_fn[tool] += 1

        if not is_match:
            errors.append({
                "name": case["name"],
                "transcript": case["transcript"][:80],
                "expected": sorted(expected),
                "predicted": sorted(predicted),
                "raw": output[:200],
            })

        if verbose:
            mark = "✓" if is_match else "✗"
            print(f"  {mark} [{i+1}/{total}] {case['name']}")
            if not is_match:
                print(f"      exp: {sorted(expected)}  got: {sorted(predicted)}")

    accuracy = exact / max(1, total)
    avg_latency = sum(latencies) / max(1, len(latencies))

    return {
        "accuracy": accuracy,
        "exact": exact,
        "total": total,
        "avg_latency_ms": round(avg_latency * 1000),
        "tool_tp": dict(tool_tp),
        "tool_fp": dict(tool_fp),
        "tool_fn": dict(tool_fn),
        "errors": errors,
    }


# ── Analysis + display ────────────────────────────────────────────

def print_analysis(
    res: dict[str, Any],
    tag: str = "",
    baseline_acc: float | None = None,
    best_acc: float = 0.0,
) -> None:
    """Print compact, actionable analysis."""
    label = f" ({tag})" if tag else ""

    print(f"\n{'━' * 60}")
    print(f"  RESULTS{label}")
    print(f"{'━' * 60}")

    # Single metric + comparisons
    acc_line = f"  Accuracy:  {res['accuracy']:.1%}  ({res['exact']}/{res['total']})"
    if baseline_acc is not None:
        delta = res["accuracy"] - baseline_acc
        acc_line += f"  (baseline: {baseline_acc:.1%}  Δ{delta:+.1%})"
    print(acc_line)
    if best_acc > 0:
        delta_best = res["accuracy"] - best_acc
        print(f"  vs best:   {best_acc:.1%}  (Δ{delta_best:+.1%})")
    print(f"  Latency:   {res['avg_latency_ms']}ms avg")

    # Per-tool table
    all_tools = set(res["tool_tp"]) | set(res["tool_fp"]) | set(res["tool_fn"])
    if all_tools:
        print(f"\n  {'Tool':<28s}  {'TP':>3s}  {'FP':>3s}  {'FN':>3s}  {'Prec':>5s}  {'Rec':>5s}")
        print(f"  {'─' * 55}")
        for tool in sorted(all_tools):
            tp = res["tool_tp"].get(tool, 0)
            fp = res["tool_fp"].get(tool, 0)
            fn = res["tool_fn"].get(tool, 0)
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            flag = " ⚠" if rec < 0.7 or prec < 0.7 else ""
            print(f"  {tool:<28s}  {tp:3d}  {fp:3d}  {fn:3d}  {prec:5.0%}  {rec:5.0%}{flag}")

    # Error details grouped by pattern
    n_err = len(res["errors"])
    if n_err > 0:
        print(f"\n  ERRORS ({n_err}):")
        patterns: dict[str, list[dict]] = defaultdict(list)
        for e in res["errors"]:
            key = f"{e['expected']} → {e['predicted']}"
            patterns[key].append(e)

        for pattern, errs in sorted(patterns.items(), key=lambda x: -len(x[1])):
            print(f"\n  ┌─ {pattern}  ({len(errs)}×)")
            for e in errs[:5]:
                print(f"  │  \"{e['transcript']}\"")
            if len(errs) > 5:
                print(f"  │  ... and {len(errs) - 5} more")
            print(f"  └")

    # Decision guidance
    print(f"\n  {'─' * 55}")
    if res["accuracy"] >= 0.95:
        print(f"  ✓ TARGET MET — accuracy ≥95%.")
    elif n_err > 0:
        pattern_counts = Counter()
        for e in res["errors"]:
            pattern_counts[f"{e['expected']}→{e['predicted']}"] += 1
        top_pattern, top_count = pattern_counts.most_common(1)[0]
        if top_count >= 3:
            print(f"  ⚠ SYSTEMATIC — \"{top_pattern}\" repeats {top_count}×.")
            print(f"    → Likely a DATA problem. Check labels for this pattern.")
        else:
            print(f"  ⚠ SCATTERED errors — need more training iters or data.")
    else:
        print(f"  ✓ No errors.")
    print(f"{'━' * 60}")


# ── Keep / discard decision ───────────────────────────────────────

def keep_or_discard(accuracy: float, best_so_far: float) -> str:
    """Decide whether to keep or discard this experiment.

    Returns "keep" or "discard".
    Simple rule: strictly better accuracy → keep.
    """
    if accuracy > best_so_far:
        return "keep"
    return "discard"


# ── Read current train hyperparams for logging ────────────────────

def _read_train_hyperparams() -> dict[str, Any]:
    """Read current hyperparams from train_multitool.py module constants."""
    from .train_multitool import (
        MULTITOOL_ITERS,
        MULTITOOL_LR,
        MULTITOOL_LORA_RANK,
        MULTITOOL_LORA_LAYERS,
    )
    return {
        "iters": MULTITOOL_ITERS,
        "lr": MULTITOOL_LR,
        "lora_rank": MULTITOOL_LORA_RANK,
        "lora_layers": MULTITOOL_LORA_LAYERS,
    }


# ── One experiment cycle ──────────────────────────────────────────

def run_one_experiment(
    iters: int,
    lr: float,
    resume: bool,
    verbose: bool,
    note: str,
    skip_train: bool = False,
    is_baseline: bool = False,
) -> dict[str, Any]:
    """
    One full experiment cycle: train → eval → log → keep/discard.

    Returns the eval results dict augmented with status.
    """
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    total_start = time.perf_counter()

    baseline_acc = _load_baseline()
    best_acc = _read_best_accuracy()
    hp = _read_train_hyperparams()

    # ── Baseline mode ──
    if is_baseline:
        print(f"\n{'━' * 60}")
        print(f"  BASELINE EVAL (no adapter)")
        print(f"{'━' * 60}")
        res = eval_adapter(None, verbose=verbose)
        _save_baseline(res["accuracy"], res)
        print_analysis(res, tag="baseline", baseline_acc=None, best_acc=best_acc)
        _append_result({
            "timestamp": run_id, "run_id": run_id,
            "iters": 0, "lr": 0, "lora_rank": 0, "lora_layers": 0,
            "accuracy": f"{res['accuracy']:.4f}",
            "exact": res["exact"], "total": res["total"],
            "elapsed_s": round(time.perf_counter() - total_start),
            "status": "baseline", "note": note or "baseline eval",
        })
        print(f"\n  Baseline cached: {res['accuracy']:.1%}")
        return res

    # ── Train ──
    if not skip_train:
        print(f"\n{'━' * 60}")
        print(f"  EXPERIMENT {run_id}: {iters} iters, lr={lr}")
        print(f"{'━' * 60}")
        try:
            train_time = quick_train(iters=iters, lr=lr, resume=resume)
            print(f"\n  Training: {train_time:.0f}s")
        except Exception as e:
            print(f"\n  ✗ TRAINING CRASHED: {e}")
            _append_result({
                "timestamp": run_id, "run_id": run_id,
                "iters": iters, "lr": lr,
                "lora_rank": hp["lora_rank"], "lora_layers": hp["lora_layers"],
                "accuracy": "0.0000", "exact": 0, "total": 0,
                "elapsed_s": round(time.perf_counter() - total_start),
                "status": "crash", "note": note or str(e)[:80],
            })
            return {"accuracy": 0.0, "status": "crash"}

    # ── Eval ──
    tag = f"{iters} iters" if not skip_train else "eval-only"
    res = eval_adapter(ADAPTER_PATH, verbose=verbose)

    # ── Decide ──
    status = keep_or_discard(res["accuracy"], best_acc) if not skip_train else "eval-only"

    if status == "keep":
        _snapshot_best(res["accuracy"])
        print(f"\n  ✓ KEEP — new best: {res['accuracy']:.1%} (was {best_acc:.1%})")
    elif status == "discard":
        print(f"\n  ✗ DISCARD — {res['accuracy']:.1%} ≤ best {best_acc:.1%}")

    # ── Log ──
    _append_result({
        "timestamp": run_id, "run_id": run_id,
        "iters": iters, "lr": lr,
        "lora_rank": hp["lora_rank"], "lora_layers": hp["lora_layers"],
        "accuracy": f"{res['accuracy']:.4f}",
        "exact": res["exact"], "total": res["total"],
        "elapsed_s": round(time.perf_counter() - total_start),
        "status": status, "note": note,
    })

    # ── Display ──
    print_analysis(res, tag=tag, baseline_acc=baseline_acc, best_acc=best_acc)

    total = time.perf_counter() - total_start
    print(f"  Experiment time: {total:.0f}s")
    res["status"] = status
    return res


# ── Print experiment history ──────────────────────────────────────

def print_history() -> None:
    """Print the results.tsv as a formatted table."""
    if not RESULTS_TSV.exists():
        print("  No experiments yet.")
        return

    print(f"\n{'━' * 80}")
    print(f"  EXPERIMENT HISTORY")
    print(f"{'━' * 80}")
    print(f"  {'Run ID':<17s} {'Iters':>5s} {'LR':>8s} {'Rank':>4s} {'Acc':>7s} {'Status':<8s} Note")
    print(f"  {'─' * 75}")

    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            acc = float(row.get("accuracy", 0))
            mark = "✓" if row.get("status") == "keep" else " "
            print(
                f"  {mark} {row.get('run_id','?'):<15s} "
                f"{row.get('iters','?'):>5s} "
                f"{row.get('lr','?'):>8s} "
                f"{row.get('lora_rank','?'):>4s} "
                f"{acc:>6.1%} "
                f"{row.get('status','?'):<8s} "
                f"{row.get('note','')}"
            )
    print(f"{'━' * 80}\n")


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoresearch experiment loop: train → eval → keep/discard → log",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--iters", type=int, default=50,
                        help="Training iterations per experiment (default: 50)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, eval current adapter")
    parser.add_argument("--baseline", action="store_true",
                        help="Eval base model without adapter, cache result")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-case pass/fail during eval")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing adapter checkpoint")
    parser.add_argument("--note", type=str, default="",
                        help="Note to attach to this experiment in results.tsv")
    parser.add_argument("--loop", type=int, default=1,
                        help="Number of experiment cycles to run (default: 1)")
    parser.add_argument("--history", action="store_true",
                        help="Print experiment history and exit")
    args = parser.parse_args()

    # ── History ──
    if args.history:
        print_history()
        return

    # ── Ensure baseline exists ──
    if not args.baseline and _load_baseline() is None:
        print("  No baseline cached. Running baseline eval first...")
        run_one_experiment(
            iters=0, lr=0, resume=False, verbose=False,
            note="auto-baseline", skip_train=True, is_baseline=True,
        )
        print()

    # ── Run experiments ──
    for cycle in range(args.loop):
        if args.loop > 1:
            print(f"\n  ═══ CYCLE {cycle + 1}/{args.loop} ═══")

        res = run_one_experiment(
            iters=args.iters,
            lr=args.lr,
            resume=args.resume,
            verbose=args.verbose,
            note=args.note,
            skip_train=args.eval_only or args.baseline,
            is_baseline=args.baseline,
        )

        # In baseline mode, only run once
        if args.baseline:
            break

    # ── Show history at the end ──
    if args.loop > 1 or not (args.eval_only or args.baseline):
        print_history()


if __name__ == "__main__":
    main()
