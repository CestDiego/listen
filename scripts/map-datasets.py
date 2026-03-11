"""
Map downloaded public datasets to the multi-tool classifier's training format.

Reads CLINC-OOS, MASSIVE, Banking77, and Snips datasets and maps their labels
to the new accommodator-based tool schema:

   - wellbeing.check_in       (kept — core safety skill)
   - accommodator.activate    (user wants mood-adaptive music/environment)
   - accommodator.skip        (user wants to skip/next track)
   - accommodator.deactivate  (user wants to stop mood accommodation)
   - accommodator.set_target  (user explicitly states desired mood/preference)

Old music.* tools are REMOVED as direct classifier targets. Music control
becomes internal to the Accommodator.

Usage:
  cd experts/ && uv run python ../scripts/map-datasets.py
  # or from project root:
  cd experts/ && uv run python -c "exec(open('../scripts/map-datasets.py').read())"
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
EXPERTS_DIR = PROJECT_ROOT / "experts"
EXPERTS_DATA_DIR = EXPERTS_DIR / "data"

OUTPUT_DIR = EXPERTS_DATA_DIR / "multitool-augmented"
EXISTING_MULTITOOL_DIR = EXPERTS_DATA_DIR / "multitool"

DATASET_DIRS = {
    "clinc-oos": DATA_DIR / "clinc-oos",
    "massive": DATA_DIR / "massive",
    "banking77": DATA_DIR / "banking77",
    "snips": DATA_DIR / "snips",
}

SEED = 42


# ── New tool definitions (accommodator pivot) ────────────────────────

NEW_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "wellbeing.check_in",
            "description": (
                "Triggered when the user expresses first-person negative self-talk, "
                "burnout, self-doubt, imposter syndrome, or emotional distress. "
                "NOT for third-person reports or casual tiredness."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accommodator.activate",
            "description": (
                "User wants mood-adaptive music or environment adjustment. "
                "Covers requests to play music, start listening, or engage the "
                "mood-responsive system."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accommodator.skip",
            "description": (
                "User wants to skip the current track without stopping or "
                "restarting the mood system. Covers: 'skip this song', "
                "'next track', 'play the next one'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accommodator.deactivate",
            "description": (
                "User wants to stop mood accommodation. Covers requests to stop, "
                "pause, or turn off the adaptive music/environment system."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accommodator.set_target",
            "description": (
                "User explicitly states a desired mood or preference. "
                "Examples: 'I want to feel calm', 'help me focus', 'I like rock music', "
                "'something upbeat please'. NOT for song info queries or playback control."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ── System prompt (new tool schema) ─────────────────────────────────

def build_system_prompt() -> str:
    """Build system prompt with the new accommodator tool definitions."""
    tool_lines = []
    for t in NEW_TOOL_DEFINITIONS:
        fn = t["function"]
        tool_lines.append(f"- {fn['name']}: {fn['description']}")
    tools_block = "\n".join(tool_lines)

    return (
        "You are a voice assistant that listens to ambient speech and activates tools "
        "when the user expresses clear intent.\n\n"
        "Available tools:\n"
        f"{tools_block}\n\n"
        "Rules:\n"
        "- Call tools ONLY for clear, direct user intent or first-person emotional distress.\n"
        "- Do NOT call tools when someone is merely talking ABOUT a topic.\n"
        "- Do NOT call tools for song lyrics, poetry, or ambient audio.\n"
        "- For wellbeing: ONLY activate for first-person self-directed distress, "
        "not third-person reports, casual tiredness, or reading/quoting others.\n"
        "- You may call MULTIPLE tools if the transcript contains multiple intents.\n"
        "- ONLY call tools from the list above. Do not invent new tools.\n"
        "- If no tools should fire, respond with: No tools needed.\n"
    )


SYSTEM_PROMPT = build_system_prompt()


# ── Tool call formatting (matches generate_multitool.py) ────────────

def format_tool_call(function_name: str) -> str:
    """Format a single tool call in Qwen3.5 native format."""
    return f"<tool_call>\n<function={function_name}>\n</function>\n</tool_call>"


def format_tool_calls(calls: list[str]) -> str:
    """Format multiple tool calls. Empty list -> 'No tools needed.'"""
    if not calls:
        return "No tools needed."
    return "\n".join(format_tool_call(name) for name in calls)


def make_entry(transcript: str, tool_calls: list[str]) -> dict:
    """Create a chat-format training entry."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": format_tool_calls(tool_calls)},
        ]
    }


# ── Label mapping rules ─────────────────────────────────────────────

# Return value: list of tool names, or None to SKIP the example entirely.

def map_clinc_label(label: str) -> list[str] | None:
    """Map CLINC-OOS label to our tool schema."""
    if label == "play_music":
        return ["accommodator.activate"]
    if label == "next_song":
        return ["accommodator.skip"]  # Fixed: was incorrectly mapped to activate
    if label == "previous_song":
        return ["accommodator.skip"]  # Track navigation = skip
    if label == "change_volume":
        return None  # SKIP — internal to accommodator
    if label == "oos":
        return []  # NEGATIVE
    # Everything else -> NEGATIVE
    return []


def map_massive_label(label: str) -> list[str] | None:
    """Map MASSIVE label to our tool schema."""
    if label == "play_music":
        return ["accommodator.activate"]
    if label in ("audio_volume_up", "audio_volume_down", "audio_volume_mute",
                 "audio_volume_other"):
        return None  # SKIP — internal to accommodator
    if label in ("music_query", "music_settings"):
        return None  # SKIP — info queries and playback control, not mood targets
    if label in ("music_likeness", "music_dislikeness"):
        return ["accommodator.set_target"]  # Preference statements are valid targets
    # Everything else -> NEGATIVE
    return []


def map_banking77_label(label: str) -> list[str] | None:
    """Map Banking77 label — ALL are negatives."""
    return []


def map_snips_label(label: str) -> list[str] | None:
    """Map Snips label to our tool schema."""
    # This dataset doesn't have PlayMusic or AddToPlaylist in our download.
    # All labels are location/restaurant/weather — pure negatives.
    return []


DATASET_MAPPERS = {
    "clinc-oos": map_clinc_label,
    "massive": map_massive_label,
    "banking77": map_banking77_label,
    "snips": map_snips_label,
}


# ── Load and map ────────────────────────────────────────────────────

def load_dataset_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file with {text, label} entries."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_existing_transcripts() -> set[str]:
    """Load existing training transcripts for dedup."""
    existing = set()
    for fname in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
        path = EXISTING_MULTITOOL_DIR / fname
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        text = d["messages"][1]["content"].lower().strip()
                        existing.add(text)
    return existing


def process_all_datasets() -> tuple[list[dict], dict]:
    """
    Process all datasets and return (entries, stats).
    
    Returns mapped training entries and detailed stats dict.
    """
    existing_texts = load_existing_transcripts()
    print(f"Loaded {len(existing_texts)} existing transcripts for dedup\n")

    all_entries: list[dict] = []
    stats: dict = {
        "per_source": {},
        "per_tool": Counter(),
        "skipped": Counter(),
        "deduped": Counter(),
    }

    for dataset_name, dataset_dir in DATASET_DIRS.items():
        mapper = DATASET_MAPPERS[dataset_name]
        source_stats = {
            "total_raw": 0,
            "mapped_positive": 0,
            "mapped_negative": 0,
            "skipped": 0,
            "deduped": 0,
            "files_processed": [],
        }

        # Process all available splits from this dataset
        for fname in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
            fpath = dataset_dir / fname
            if not fpath.exists():
                continue

            source_stats["files_processed"].append(fname)
            raw_entries = load_dataset_jsonl(fpath)
            source_stats["total_raw"] += len(raw_entries)

            for raw in raw_entries:
                text = raw["text"]
                label = raw["label"]

                # Map label
                tool_calls = mapper(label)
                if tool_calls is None:
                    source_stats["skipped"] += 1
                    stats["skipped"][f"{dataset_name}/{label}"] += 1
                    continue

                # Dedup against existing training data
                text_key = text.lower().strip()
                if text_key in existing_texts:
                    source_stats["deduped"] += 1
                    stats["deduped"][dataset_name] += 1
                    continue

                # Dedup within new data (track seen texts)
                if text_key in existing_texts:
                    continue
                existing_texts.add(text_key)

                # Create entry
                entry = make_entry(text, tool_calls)
                entry["_source"] = dataset_name
                entry["_label"] = label
                entry["_tools"] = tool_calls
                all_entries.append(entry)

                if tool_calls:
                    source_stats["mapped_positive"] += 1
                    for tc in tool_calls:
                        stats["per_tool"][tc] += 1
                else:
                    source_stats["mapped_negative"] += 1
                    stats["per_tool"]["NEGATIVE"] += 1

        stats["per_source"][dataset_name] = source_stats
        print(f"  {dataset_name}:")
        print(f"    Raw: {source_stats['total_raw']}, "
              f"Pos: {source_stats['mapped_positive']}, "
              f"Neg: {source_stats['mapped_negative']}, "
              f"Skip: {source_stats['skipped']}, "
              f"Dedup: {source_stats['deduped']}")

    return all_entries, stats


# ── Balancing ────────────────────────────────────────────────────────

def balance_dataset(
    entries: list[dict],
    target_positive_ratio: float = 0.40,
    rng: random.Random | None = None,
) -> list[dict]:
    """
    Balance to ~40% positive / ~60% negative by subsampling negatives.
    
    Never subsamples positives (especially wellbeing — safety-critical).
    """
    if rng is None:
        rng = random.Random(SEED)

    positives = [e for e in entries if e["_tools"]]
    negatives = [e for e in entries if not e["_tools"]]

    n_pos = len(positives)
    if n_pos == 0:
        print("  WARNING: No positive examples found!")
        return entries

    # How many negatives to keep for target ratio
    # target_pos_ratio = n_pos / (n_pos + n_neg_target)
    # => n_neg_target = n_pos * (1 - target_pos_ratio) / target_pos_ratio
    n_neg_target = int(n_pos * (1 - target_positive_ratio) / target_positive_ratio)

    if len(negatives) <= n_neg_target:
        print(f"  Balancing: keeping all {len(negatives)} negatives "
              f"(target was {n_neg_target})")
        return entries

    # Subsample negatives, stratified by source to maintain diversity
    neg_by_source: dict[str, list[dict]] = defaultdict(list)
    for e in negatives:
        neg_by_source[e["_source"]].append(e)

    # Proportional allocation per source
    sampled_negatives: list[dict] = []
    total_neg = len(negatives)

    for source, source_negs in neg_by_source.items():
        proportion = len(source_negs) / total_neg
        n_from_source = max(1, int(n_neg_target * proportion))
        rng.shuffle(source_negs)
        sampled_negatives.extend(source_negs[:n_from_source])

    # Trim to exact target if proportional allocation overshot
    if len(sampled_negatives) > n_neg_target:
        rng.shuffle(sampled_negatives)
        sampled_negatives = sampled_negatives[:n_neg_target]

    print(f"  Balancing: {n_pos} positives, "
          f"{len(negatives)} -> {len(sampled_negatives)} negatives "
          f"(target ratio: {target_positive_ratio:.0%})")

    result = positives + sampled_negatives
    rng.shuffle(result)
    return result


# ── Splitting ────────────────────────────────────────────────────────

def stratified_split(
    entries: list[dict],
    ratios: tuple[float, float, float] = (0.80, 0.10, 0.10),
    rng: random.Random | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified split by tool category.
    
    Ensures each split has proportional representation of each tool type.
    """
    if rng is None:
        rng = random.Random(SEED)

    # Group by tool signature
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        key = ",".join(sorted(e["_tools"])) if e["_tools"] else "NEGATIVE"
        by_cat[key].append(e)

    train: list[dict] = []
    valid: list[dict] = []
    test: list[dict] = []

    for cat, items in by_cat.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * ratios[0]))
        n_valid = max(1, int(n * ratios[1]))
        train.extend(items[:n_train])
        valid.extend(items[n_train:n_train + n_valid])
        test.extend(items[n_train + n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)

    return train, valid, test


# ── Output ───────────────────────────────────────────────────────────

def write_jsonl(entries: list[dict], path: Path) -> None:
    """Write entries as JSONL, stripping internal metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            # Strip internal metadata before writing
            clean = {
                "messages": entry["messages"]
            }
            f.write(json.dumps(clean, separators=(",", ":")) + "\n")


def compute_final_stats(
    train: list[dict],
    valid: list[dict],
    test: list[dict],
    process_stats: dict,
) -> dict:
    """Compute comprehensive stats for the output."""
    stats = {
        "source_stats": process_stats["per_source"],
        "per_tool_total": dict(process_stats["per_tool"]),
        "skipped_labels": dict(process_stats["skipped"]),
        "deduped_per_source": dict(process_stats["deduped"]),
        "splits": {},
    }

    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        tool_counts: Counter = Counter()
        source_counts: Counter = Counter()

        for e in split:
            if e["_tools"]:
                for tc in e["_tools"]:
                    tool_counts[tc] += 1
            else:
                tool_counts["NEGATIVE"] += 1
            source_counts[e["_source"]] += 1

        n_pos = sum(1 for e in split if e["_tools"])
        n_total = len(split)

        stats["splits"][name] = {
            "total": n_total,
            "positive": n_pos,
            "negative": n_total - n_pos,
            "positive_ratio": round(n_pos / n_total, 3) if n_total else 0,
            "per_tool": dict(tool_counts),
            "per_source": dict(source_counts),
        }

    # Grand totals
    total = len(train) + len(valid) + len(test)
    total_pos = sum(s["positive"] for s in stats["splits"].values())
    stats["grand_total"] = {
        "total": total,
        "positive": total_pos,
        "negative": total - total_pos,
        "positive_ratio": round(total_pos / total, 3) if total else 0,
    }

    return stats


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'=' * 70}")
    print("Mapping public datasets to multi-tool training format")
    print(f"  Tool schema: accommodator.activate, accommodator.skip,")
    print(f"               accommodator.deactivate, accommodator.set_target,")
    print(f"               wellbeing.check_in")
    print(f"{'=' * 70}\n")

    # 1. Process all datasets
    print("Step 1: Loading and mapping datasets...")
    entries, process_stats = process_all_datasets()

    total_pos = sum(1 for e in entries if e["_tools"])
    total_neg = sum(1 for e in entries if not e["_tools"])
    print(f"\n  Total mapped: {len(entries)} "
          f"(pos: {total_pos}, neg: {total_neg})\n")

    # 2. Balance
    print("Step 2: Balancing dataset...")
    rng = random.Random(SEED)
    balanced = balance_dataset(entries, target_positive_ratio=0.40, rng=rng)

    bal_pos = sum(1 for e in balanced if e["_tools"])
    bal_neg = sum(1 for e in balanced if not e["_tools"])
    print(f"  After balancing: {len(balanced)} "
          f"(pos: {bal_pos}, neg: {bal_neg}, "
          f"ratio: {bal_pos / len(balanced):.1%})\n")

    # 3. Split
    print("Step 3: Stratified split (80/10/10)...")
    train, valid, test = stratified_split(balanced, rng=rng)

    # 4. Write output
    print("\nStep 4: Writing output files...")
    write_jsonl(train, OUTPUT_DIR / "train.jsonl")
    write_jsonl(valid, OUTPUT_DIR / "valid.jsonl")
    write_jsonl(test, OUTPUT_DIR / "test.jsonl")

    # 5. Compute and write stats
    final_stats = compute_final_stats(train, valid, test, process_stats)
    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(final_stats, f, indent=2)

    # 6. Print summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\nPer-source contributions:")
    for source, ss in final_stats["source_stats"].items():
        print(f"  {source:12s}: {ss['mapped_positive']:5d} pos, "
              f"{ss['mapped_negative']:5d} neg, "
              f"{ss['skipped']:4d} skip, "
              f"{ss['deduped']:4d} dedup")

    print(f"\nPer-tool counts (before split):")
    for tool, count in sorted(process_stats["per_tool"].items()):
        print(f"  {tool:30s}: {count:5d}")

    print(f"\nSkipped labels (volume controls, internal):")
    for label, count in sorted(process_stats["skipped"].items()):
        print(f"  {label:40s}: {count:4d}")

    print(f"\nPer-split breakdown:")
    for name in ["train", "valid", "test"]:
        s = final_stats["splits"][name]
        print(f"  {name:5s}: {s['total']:5d} total "
              f"({s['positive']:4d} pos, {s['negative']:4d} neg, "
              f"ratio: {s['positive_ratio']:.1%})")
        for tool, count in sorted(s["per_tool"].items()):
            print(f"         {tool:30s}: {count:4d}")

    gt = final_stats["grand_total"]
    print(f"\n  GRAND TOTAL: {gt['total']} examples "
          f"({gt['positive']} pos, {gt['negative']} neg, "
          f"ratio: {gt['positive_ratio']:.1%})")

    print(f"\nOutput files:")
    for fname in ["train.jsonl", "valid.jsonl", "test.jsonl", "stats.json"]:
        p = OUTPUT_DIR / fname
        size_kb = p.stat().st_size / 1024
        print(f"  {p}  ({size_kb:.1f} KB)")

    print(f"\n{'=' * 70}")
    print("Done. Dataset is ready for training.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
