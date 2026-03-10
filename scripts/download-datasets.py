#!/usr/bin/env python3
"""Download public HuggingFace datasets for multi-tool classifier augmentation.

Datasets:
  1. CLINC150-OOS (plus variant) — 23.9k utterances, 150 intents + out-of-scope
  2. Amazon MASSIVE en-US — 11.5k train, 60 intents
  3. Snips built-in intents — NLU benchmark
  4. BANKING77 — 13k utterances, 77 intents (pure negatives for Listen)

Usage:
    uv run python scripts/download-datasets.py          # from project root
    cd experts && uv run python ../scripts/download-datasets.py

Only downloads datasets whose target directory doesn't already contain JSONL files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

TARGETS = {
    "clinc-oos": DATA_DIR / "clinc-oos",
    "massive": DATA_DIR / "massive",
    "snips": DATA_DIR / "snips",
    "banking77": DATA_DIR / "banking77",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dir_has_jsonl(p: Path) -> bool:
    """Return True if directory exists and contains at least one .jsonl file."""
    return p.is_dir() and any(p.glob("*.jsonl"))


def resolve_labels(ds_split, label_col: str = "label") -> list[str] | None:
    """Return list of label name strings if label_col is ClassLabel, else None."""
    feat = ds_split.features.get(label_col)
    if hasattr(feat, "names"):
        return feat.names
    return None


def save_split_jsonl(
    dataset_dict,
    out_dir: Path,
    *,
    text_col: str = "text",
    label_col: str = "label",
    label_text_col: str | None = None,
    extra_cols: dict[str, str] | None = None,
) -> dict:
    """Save every split as JSONL with normalized schema {text, label, ...}.

    If label_text_col is given, use that column directly as the string label.
    Otherwise, resolve ClassLabel int → name automatically.
    extra_cols maps output_key → source_column for additional fields.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}

    for split_name, ds in dataset_dict.items():
        label_names = resolve_labels(ds, label_col)
        path = out_dir / f"{split_name}.jsonl"
        labels_seen: list[str] = []

        with open(path, "w") as f:
            for row in ds:
                text = row[text_col]

                if label_text_col and label_text_col in row:
                    label_str = row[label_text_col]
                elif label_names is not None:
                    label_str = label_names[row[label_col]]
                else:
                    label_str = str(row[label_col])

                record: dict = {"text": text, "label": label_str}
                if extra_cols:
                    for out_key, src_key in extra_cols.items():
                        record[out_key] = row[src_key]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                labels_seen.append(label_str)

        label_counts = Counter(labels_seen)
        stats[split_name] = {
            "examples": len(ds),
            "unique_labels": len(label_counts),
            "top5": label_counts.most_common(5),
        }
    return stats


def human_size(path: Path) -> str:
    """Return human-readable size of a directory tree."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.1f} {unit}"
        total /= 1024
    return f"{total:.1f} TB"


def print_stats(name: str, stats: dict, out_dir: Path) -> None:
    total_ex = sum(s["examples"] for s in stats.values())
    unique_labels = max((s["unique_labels"] for s in stats.values()), default=0)
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  Path: {out_dir}")
    print(f"  Disk: {human_size(out_dir)}")
    print(f"  Total examples: {total_ex:,}  |  Unique labels: {unique_labels}")
    for split, s in stats.items():
        top = ", ".join(f"{lbl}({n})" for lbl, n in s["top5"][:3])
        print(f"    {split:12s}  {s['examples']:>6,} examples  —  top: {top}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------

def download_clinc(out_dir: Path) -> None:
    print("\n[1/4] Downloading CLINC150-OOS (plus variant)...")
    ds = load_dataset("clinc/clinc_oos", "plus")
    # Features: text (str), intent (ClassLabel with 151 names including oos)
    stats = save_split_jsonl(ds, out_dir, label_col="intent")
    print_stats("CLINC150-OOS (plus)", stats, out_dir)


def download_massive(out_dir: Path) -> None:
    print("\n[2/4] Downloading Amazon MASSIVE (en-US only)...")
    # Official AmazonScience/massive uses a legacy loading script unsupported
    # by datasets>=4.  Use the SetFit community mirror instead.
    ds = load_dataset("SetFit/amazon_massive_intent_en-US")
    # Columns: id, label (ClassLabel int), text, label_text (str)
    stats = save_split_jsonl(ds, out_dir, label_text_col="label_text")
    print_stats("MASSIVE en-US", stats, out_dir)


def download_snips(out_dir: Path) -> None:
    print("\n[3/4] Downloading Snips built-in intents...")
    ds = load_dataset("snips_built_in_intents")
    # Columns: text (str), label (ClassLabel)
    stats = save_split_jsonl(ds, out_dir)
    print_stats("Snips built-in intents", stats, out_dir)


def download_banking77(out_dir: Path) -> None:
    print("\n[4/4] Downloading BANKING77...")
    # PolyAI/banking77 uses a legacy script; the community copy works.
    ds = load_dataset("banking77")
    # Columns: text (str), label (ClassLabel with 77 names)
    stats = save_split_jsonl(ds, out_dir)
    print_stats("BANKING77", stats, out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DOWNLOADERS = [
    ("clinc-oos", download_clinc),
    ("massive", download_massive),
    ("snips", download_snips),
    ("banking77", download_banking77),
]


def main() -> None:
    print(f"Data directory: {DATA_DIR}")
    skipped = 0
    for name, fn in DOWNLOADERS:
        out = TARGETS[name]
        if dir_has_jsonl(out):
            print(f"\n  SKIP {name} — already has JSONL files in {out}")
            skipped += 1
            continue
        fn(out)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_disk = 0
    for name, out in TARGETS.items():
        if out.is_dir():
            size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
            total_disk += size
            print(f"  {name:12s}  {human_size(out):>10s}  {out}")
        else:
            print(f"  {name:12s}  {'(missing)':>10s}  {out}")
    print(f"\n  Total disk usage: {total_disk / 1024 / 1024:.1f} MB")
    if skipped:
        print(f"  ({skipped} dataset(s) skipped — already present)")
    print()


if __name__ == "__main__":
    main()
