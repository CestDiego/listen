"""
Merge training data: hand-crafted + augmented → deduplicated, rebalanced, split.

1. Re-generates the hand-crafted data (calls generate_multitool_data())
2. Loads the augmented data from experts/data/multitool-augmented/
3. Merges them, deduplicating by transcript text
4. Rebalances to ~40% positive / 60% negative
5. Does a stratified split (80/10/10 train/valid/test)
6. Writes to experts/data/multitool/ (overwrites the old data)
7. Prints detailed stats

Usage:
  cd /Users/diego/Projects/listen/experts && uv run python ../scripts/merge-training-data.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add experts package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "experts"))

from experts.config import DATA_DIR
from experts.generate_multitool import (
    MUSIC_TO_ACCOMMODATOR,
    generate_multitool_data,
    write_jsonl,
    _deduplicate,
    _stratified_split,
)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def remap_augmented_entry(entry: dict) -> dict:
    """Remap any old music.* tool calls in augmented data to accommodator.* tools."""
    messages = entry["messages"]
    assistant_content = messages[-1]["content"]

    # Check if this entry contains old music.* tool calls
    needs_remap = False
    for old_action in MUSIC_TO_ACCOMMODATOR:
        if f"music.{old_action}" in assistant_content:
            needs_remap = True
            break

    if not needs_remap:
        return entry

    # Remap each old music.* reference
    new_content = assistant_content
    for old_action, new_tool in MUSIC_TO_ACCOMMODATOR.items():
        new_content = new_content.replace(f"music.{old_action}", new_tool)

    # Also remap in system prompt if present
    new_messages = []
    for msg in messages:
        new_messages.append(dict(msg))
    new_messages[-1]["content"] = new_content

    return {"messages": new_messages}


def count_stats(entries: list[dict]) -> dict:
    """Compute detailed stats for a set of entries."""
    total = len(entries)
    tool_counter: Counter = Counter()
    n_positive = 0
    n_negative = 0
    n_dual = 0

    for e in entries:
        content = e["messages"][-1]["content"]
        n_tools = content.count("<tool_call>")
        if n_tools == 0:
            n_negative += 1
        else:
            n_positive += 1
            if n_tools >= 2:
                n_dual += 1

        # Extract tool names
        import re
        for match in re.finditer(r"<function=([\w.]+)>", content):
            tool_counter[match.group(1)] += 1

    return {
        "total": total,
        "positive": n_positive,
        "negative": n_negative,
        "dual": n_dual,
        "positive_ratio": n_positive / max(1, total),
        "per_tool": dict(tool_counter.most_common()),
    }


def rebalance(
    entries: list[dict],
    target_positive_ratio: float = 0.40,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Rebalance entries to target ~40% positive / 60% negative.

    If we have too many positives, we downsample positives.
    If we have too many negatives, we downsample negatives.
    """
    if rng is None:
        rng = random.Random(42)

    positives = [e for e in entries if "<tool_call>" in e["messages"][-1]["content"]]
    negatives = [e for e in entries if "<tool_call>" not in e["messages"][-1]["content"]]

    n_pos = len(positives)
    n_neg = len(negatives)
    total = n_pos + n_neg

    current_ratio = n_pos / max(1, total)

    if abs(current_ratio - target_positive_ratio) < 0.05:
        # Already close enough
        return entries

    if current_ratio > target_positive_ratio:
        # Too many positives — downsample them
        target_pos = int(n_neg * target_positive_ratio / (1 - target_positive_ratio))
        target_pos = min(target_pos, n_pos)  # Don't upsample
        rng.shuffle(positives)
        positives = positives[:target_pos]
    else:
        # Too many negatives — downsample them
        target_neg = int(n_pos * (1 - target_positive_ratio) / target_positive_ratio)
        target_neg = min(target_neg, n_neg)  # Don't upsample
        rng.shuffle(negatives)
        negatives = negatives[:target_neg]

    result = positives + negatives
    rng.shuffle(result)
    return result


def main() -> None:
    print(f"\n{'='*60}")
    print("MERGE TRAINING DATA: hand-crafted + augmented")
    print(f"{'='*60}")

    rng = random.Random(42)

    # 1. Generate hand-crafted data
    print("\n[1/6] Generating hand-crafted data...")
    train_hc, valid_hc, test_hc = generate_multitool_data(seed=42)
    all_handcrafted = train_hc + valid_hc + test_hc
    hc_stats = count_stats(all_handcrafted)
    print(f"  Hand-crafted: {hc_stats['total']} entries ({hc_stats['positive']} pos, {hc_stats['negative']} neg)")

    # 2. Load augmented data
    print("\n[2/6] Loading augmented data...")
    aug_dir = DATA_DIR / "multitool-augmented"
    all_augmented = []
    for split_name in ["train", "valid", "test"]:
        path = aug_dir / f"{split_name}.jsonl"
        if path.exists():
            split_data = load_jsonl(path)
            print(f"  {split_name}: {len(split_data)} entries")
            all_augmented.extend(split_data)
        else:
            print(f"  {split_name}: not found, skipping")

    # Remap old music.* calls in augmented data
    print(f"\n[3/6] Remapping augmented data (music.* -> accommodator.*)...")
    remapped = [remap_augmented_entry(e) for e in all_augmented]
    aug_stats = count_stats(remapped)
    print(f"  Augmented (remapped): {aug_stats['total']} entries ({aug_stats['positive']} pos, {aug_stats['negative']} neg)")

    # 3. Merge + deduplicate
    print(f"\n[4/6] Merging and deduplicating...")
    merged = all_handcrafted + remapped
    print(f"  Before dedup: {len(merged)}")
    merged = _deduplicate(merged)
    print(f"  After dedup: {len(merged)}")

    # 4. Rebalance
    print(f"\n[5/6] Rebalancing to ~40% positive / 60% negative...")
    pre_balance_stats = count_stats(merged)
    print(f"  Before: {pre_balance_stats['positive_ratio']:.1%} positive")
    merged = rebalance(merged, target_positive_ratio=0.40, rng=rng)
    post_balance_stats = count_stats(merged)
    print(f"  After: {post_balance_stats['positive_ratio']:.1%} positive ({post_balance_stats['total']} entries)")

    # 5. Stratified split 80/10/10
    print(f"\n[6/6] Stratified split (80/10/10)...")
    train, valid, test = _stratified_split(merged, ratios=(0.80, 0.10, 0.10), rng=rng)

    # Write output
    out_dir = DATA_DIR / "multitool"
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(valid, out_dir / "valid.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    # Final stats
    print(f"\n{'='*60}")
    print("FINAL DATASET STATS")
    print(f"{'='*60}")

    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        stats = count_stats(split)
        print(f"\n  {name}: {stats['total']} entries")
        print(f"    Positive: {stats['positive']} ({stats['positive_ratio']:.1%})")
        print(f"    Negative: {stats['negative']}")
        print(f"    Dual-activation: {stats['dual']}")
        print(f"    Per-tool: {stats['per_tool']}")

    total_stats = count_stats(train + valid + test)
    print(f"\n  TOTAL: {total_stats['total']} entries")
    print(f"    Positive: {total_stats['positive']} ({total_stats['positive_ratio']:.1%})")
    print(f"    Negative: {total_stats['negative']}")
    print(f"    Dual-activation: {total_stats['dual']}")
    print(f"    Per-tool breakdown: {total_stats['per_tool']}")

    # Source breakdown
    print(f"\n  Sources:")
    print(f"    Hand-crafted: {hc_stats['total']}")
    print(f"    Augmented: {aug_stats['total']}")
    print(f"    After merge+dedup+rebalance: {total_stats['total']}")

    print(f"\n  Saved to: {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
