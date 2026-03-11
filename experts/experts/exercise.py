"""
Per-skill data exercises — data-centric autoresearch.

Instead of tweaking hyperparams on fixed data, we fix each skill's data
one at a time and verify with micro-experiments. Each exercise:

  1. DIAGNOSE: Show coverage, patterns, gaps for one skill
  2. GENERATE: Add targeted examples (positives + hard negatives)
  3. RETRAIN: Quick micro-experiment (50 iters)
  4. VERIFY:  Did this skill improve? Did others degrade?

This flips Karpathy's loop: the MODEL is fixed, the DATA is the variable.

Usage:
  uv run python -m experts.exercise diagnose             # all skills
  uv run python -m experts.exercise diagnose --skill skip # one skill
  uv run python -m experts.exercise add --skill deactivate --file new_examples.jsonl
  uv run python -m experts.exercise run --skill deactivate  # diagnose + retrain + verify

Philosophy:
  - Each skill is an independent exercise
  - A "cycle" = diagnose all skills → exercise the worst → verify
  - The agent picks the skill with the biggest gap
  - Never add data for ALL skills at once (can't attribute improvement)
  - After each exercise: retrain 50 iters, eval, check per-skill delta
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import DATA_DIR, TOOL_DEFINITIONS


# ── Paths ─────────────────────────────────────────────────────────

TRAIN_PATH = DATA_DIR / "multitool" / "train.jsonl"
VALID_PATH = DATA_DIR / "multitool" / "valid.jsonl"
TEST_PATH = DATA_DIR / "multitool" / "test.jsonl"
EVAL_CASES_PATH = (
    Path(__file__).parent.parent.parent
    / "src" / "listen" / "skills" / "eval-cases.json"
)

VALID_TOOLS = {t["function"]["name"] for t in TOOL_DEFINITIONS}
TOOL_CALL_RE = re.compile(r"<function=([\w.]+)>")


# ── Data loading ──────────────────────────────────────────────────

def load_training_data(path: Path | None = None) -> list[dict]:
    """Load training JSONL, extracting transcript + tool calls."""
    path = path or TRAIN_PATH
    entries = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            transcript = d["messages"][1]["content"]
            assistant = d["messages"][-1]["content"]
            tools = TOOL_CALL_RE.findall(assistant)
            entries.append({
                "transcript": transcript,
                "tools": tools,
                "is_negative": "no tools needed" in assistant.lower(),
            })
    return entries


def load_eval_cases() -> list[dict]:
    """Load curated eval cases."""
    with open(EVAL_CASES_PATH) as f:
        data = json.load(f)
    cases = []
    for c in data["cases"]:
        if "_section" in c:
            continue
        skills = c["expect"].get("skills", [])
        tools = [f"{s['skill']}.{s['action']}" for s in skills]
        cases.append({
            "name": c["name"],
            "transcript": c["transcript"],
            "tools": tools,
        })
    return cases


# ── Pattern classification ────────────────────────────────────────

# Keywords for rough pattern classification per skill
PATTERN_KEYWORDS: dict[str, list[tuple[str, list[str]]]] = {
    "accommodator.activate": [
        ("play specific", ["play .+ by", "play .+ from", "put on .+ by"]),
        ("play playlist", ["playlist", "my .+ playlist"]),
        ("play genre", ["play jazz", "play rock", "play pop", "play hip hop",
                        "play country", "play classical", "play r.?&.?b"]),
        ("play generic", ["play some music", "play music", "put on some"]),
        ("resume/unpause", ["unpause", "resume", "continue play", "keep playing",
                           "bring .+ back", "start .+ again"]),
        ("mood request", ["match my mood", "something .+ fits", "mood music",
                         "background music", "vibe"]),
        ("start/turn on", ["start the music", "turn on", "start listening"]),
    ],
    "accommodator.skip": [
        ("skip forward", ["skip", "next song", "next track", "next one",
                         "next please", "play the next", "go to the next"]),
        ("skip backward", ["previous", "go back", "last song", "replay",
                          "rewind", "before"]),
        ("change song", ["change the song", "change the track", "switch"]),
        ("dislike skip", ["don't like this", "hate this", "can't stand",
                         "tired of this"]),
    ],
    "accommodator.deactivate": [
        ("pause", ["pause"]),
        ("stop", ["stop .+ music", "stop playing", "stop accommodat"]),
        ("mute/silence", ["mute", "silence", "quiet"]),
        ("turn off", ["turn off", "turn it off", "kill the music"]),
        ("don't want", ["don't want music", "no more music", "enough"]),
    ],
    "accommodator.set_target": [
        ("mood target", ["feel calm", "feel energi", "wind down", "chill",
                        "help me focus", "concentrate", "relax", "upbeat",
                        "peaceful", "mellow"]),
        ("volume", ["turn it up", "turn it down", "volume", "louder",
                   "quieter", "softer", "too loud"]),
        ("genre preference", ["i like", "i love", "i prefer", "i'm into",
                             "i'm in the mood for"]),
        ("dislike", ["don't like", "hate", "never play", "can't stand"]),
        ("save/favorite", ["save", "favorite", "add to", "bookmark",
                          "remember this"]),
        ("rating", ["rate", "star", "thumb"]),
    ],
    "wellbeing.check_in": [
        ("self-criticism", ["stupid", "dumb", "idiot", "useless", "failure"]),
        ("worthlessness", ["worthless", "worthl", "not good enough", "don't deserve"]),
        ("burnout", ["burned out", "burning out", "exhausted", "can't take",
                    "drowning", "collapsing"]),
        ("imposter", ["fraud", "faking", "impostor", "imposter", "pretend",
                     "don't belong", "figure out", "realize"]),
        ("giving up", ["give up", "giving up", "why do i even", "what's the point",
                      "done trying", "quit everything"]),
        ("self-hate", ["hate myself", "hate how", "hate that i"]),
        ("loneliness", ["alone", "nobody cares", "no one", "isolated", "nobody notices",
                       "nobody understands", "no one to talk"]),
        ("crying/broken", ["crying", "broken", "empty", "falling apart"]),
        ("anxiety", ["panic", "anxiety", "anxious", "shaking", "spiraling",
                    "heart is racing"]),
        ("grief", ["grief", "miss them", "losing them", "reminds me of them"]),
        ("hopelessness", ["no way out", "trapped", "no hope", "never get better",
                         "future looks.*dark"]),
    ],
}


def classify_pattern(transcript: str, tool: str) -> str:
    """Classify a transcript into a pattern category for a tool."""
    text = transcript.lower()
    patterns = PATTERN_KEYWORDS.get(tool, [])
    for pattern_name, keywords in patterns:
        for kw in keywords:
            if re.search(kw, text, re.IGNORECASE):
                return pattern_name
    return "other"


# ── Diagnosis ─────────────────────────────────────────────────────

def diagnose_skill(entries: list[dict], eval_cases: list[dict], tool: str) -> dict:
    """Generate a diagnostic report for one skill."""
    # Positive examples for this tool
    positives = [e for e in entries if tool in e["tools"]]
    single = [e for e in positives if len(e["tools"]) == 1]
    dual = [e for e in positives if len(e["tools"]) > 1]

    # Pattern distribution
    pattern_counts: Counter = Counter()
    pattern_examples: dict[str, list[str]] = defaultdict(list)
    for e in single:
        pat = classify_pattern(e["transcript"], tool)
        pattern_counts[pat] += 1
        if len(pattern_examples[pat]) < 3:
            pattern_examples[pat].append(e["transcript"][:70])

    # Hard negatives: negatives that contain keywords for this tool
    tool_keywords = []
    for _, keywords in PATTERN_KEYWORDS.get(tool, []):
        tool_keywords.extend(keywords)

    hard_negatives = []
    for e in entries:
        if e["is_negative"]:
            text = e["transcript"].lower()
            for kw in tool_keywords:
                if re.search(kw, text, re.IGNORECASE):
                    hard_negatives.append(e)
                    break

    # Eval case coverage
    tool_eval_cases = [c for c in eval_cases if tool in c["tools"]]
    eval_patterns: Counter = Counter()
    for c in tool_eval_cases:
        pat = classify_pattern(c["transcript"], tool)
        eval_patterns[pat] += 1

    # Find eval patterns with few training examples
    gaps = []
    for pat, count in eval_patterns.items():
        train_count = pattern_counts.get(pat, 0)
        if train_count < 5:
            gaps.append((pat, train_count, count))

    return {
        "tool": tool,
        "total_positive": len(positives),
        "single": len(single),
        "dual": len(dual),
        "patterns": dict(pattern_counts.most_common()),
        "pattern_examples": dict(pattern_examples),
        "hard_negatives": len(hard_negatives),
        "eval_cases": len(tool_eval_cases),
        "gaps": gaps,
    }


def diagnose_negatives(entries: list[dict]) -> dict:
    """Diagnose negative example coverage."""
    negatives = [e for e in entries if e["is_negative"]]

    # Check for domain-specific hard negatives
    hard_neg_categories = {
        "music-adjacent": [
            "song", "album", "concert", "guitar", "band", "playlist",
            "chorus", "rhythm", "melody", "genre", "DJ",
        ],
        "skip-adjacent": [
            "skip", "next topic", "move on", "go ahead",
        ],
        "stop-adjacent": [
            "stop", "pause", "hold on", "wait",
        ],
        "wellbeing-adjacent": [
            "stressed", "tired", "exhausted", "depressed", "overwhelmed",
            "crying", "hate", "worthless",
        ],
    }

    category_counts: Counter = Counter()
    for e in negatives:
        text = e["transcript"].lower()
        matched = False
        for cat, keywords in hard_neg_categories.items():
            for kw in keywords:
                if kw in text:
                    category_counts[cat] += 1
                    matched = True
                    break
            if matched:
                break
        if not matched:
            category_counts["general"] += 1

    return {
        "total": len(negatives),
        "categories": dict(category_counts.most_common()),
    }


def diagnose_duals(entries: list[dict]) -> dict:
    """Diagnose dual-activation coverage."""
    duals = [e for e in entries if len(e["tools"]) >= 2]
    combos: Counter = Counter()
    combo_examples: dict[str, list[str]] = defaultdict(list)

    for e in duals:
        key = " + ".join(sorted(e["tools"]))
        combos[key] += 1
        if len(combo_examples[key]) < 2:
            combo_examples[key].append(e["transcript"][:70])

    return {
        "total": len(duals),
        "combos": dict(combos.most_common()),
        "examples": dict(combo_examples),
    }


# ── Display ───────────────────────────────────────────────────────

def print_diagnosis(skill_report: dict) -> None:
    """Print a skill diagnostic report."""
    r = skill_report
    print(f"\n  {'━' * 55}")
    print(f"  {r['tool']}")
    print(f"  {'━' * 55}")
    print(f"  Positives: {r['total_positive']}  (single: {r['single']}, dual: {r['dual']})")
    print(f"  Hard negatives: {r['hard_negatives']}")
    print(f"  Eval cases: {r['eval_cases']}")

    if r["patterns"]:
        print(f"\n  Patterns:")
        for pat, count in r["patterns"].items():
            examples = r["pattern_examples"].get(pat, [])
            ex_str = f"  e.g. \"{examples[0]}\"" if examples else ""
            print(f"    {pat:<25s}  {count:4d}{ex_str}")

    if r["gaps"]:
        print(f"\n  ⚠ Gaps (eval patterns with <5 training examples):")
        for pat, train_count, eval_count in r["gaps"]:
            print(f"    {pat:<25s}  train={train_count}  eval={eval_count}")

    # Health assessment
    health = "✅ GOOD" if r["total_positive"] >= 50 and r["hard_negatives"] >= 10 else \
             "⚠️ LOW" if r["total_positive"] >= 20 else "🔴 CRITICAL"
    if r["hard_negatives"] < 5:
        health = "🔴 CRITICAL (no hard negatives)"
    print(f"\n  Health: {health}")


def print_full_diagnosis(
    skill_reports: list[dict],
    neg_report: dict,
    dual_report: dict,
) -> None:
    """Print full diagnostic across all skills."""
    print(f"\n{'━' * 60}")
    print(f"  DATA EXERCISE DIAGNOSIS")
    print(f"{'━' * 60}")

    # Summary table
    print(f"\n  {'Tool':<28s}  {'Pos':>4s}  {'HNeg':>5s}  {'Eval':>4s}  Health")
    print(f"  {'─' * 55}")
    for r in skill_reports:
        health = "✅" if r["total_positive"] >= 50 and r["hard_negatives"] >= 10 else \
                 "⚠️" if r["total_positive"] >= 20 else "🔴"
        if r["hard_negatives"] < 5:
            health = "🔴"
        print(f"  {r['tool']:<28s}  {r['total_positive']:4d}  {r['hard_negatives']:5d}  "
              f"{r['eval_cases']:4d}  {health}")

    # Negatives
    print(f"\n  Negatives: {neg_report['total']}")
    for cat, count in neg_report["categories"].items():
        print(f"    {cat:<25s}  {count:4d}")

    # Duals
    print(f"\n  Dual-activation: {dual_report['total']}")
    for combo, count in dual_report["combos"].items():
        print(f"    {combo:<45s}  {count:2d}")

    # Per-skill details
    for r in skill_reports:
        print_diagnosis(r)

    # Exercise priority
    print(f"\n{'━' * 60}")
    print(f"  EXERCISE PRIORITY (worst first)")
    print(f"{'━' * 60}")
    priority = sorted(skill_reports, key=lambda r: (
        r["total_positive"] + r["hard_negatives"] * 2
    ))
    for i, r in enumerate(priority, 1):
        gaps_str = ", ".join(g[0] for g in r["gaps"][:3]) if r["gaps"] else "none detected"
        print(f"  {i}. {r['tool']} — {r['total_positive']} pos, "
              f"{r['hard_negatives']} hard neg. Gaps: {gaps_str}")
    print(f"{'━' * 60}\n")


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-skill data exercises for data-centric autoresearch",
    )
    sub = parser.add_subparsers(dest="command")

    diag = sub.add_parser("diagnose", help="Diagnose data coverage")
    diag.add_argument("--skill", type=str, help="Focus on one skill")
    diag.add_argument("--split", default="train", choices=["train", "valid", "test"])

    parser.parse_args()
    args = parser.parse_args()

    if args.command == "diagnose":
        path = {"train": TRAIN_PATH, "valid": VALID_PATH, "test": TEST_PATH}[args.split]
        entries = load_training_data(path)
        eval_cases = load_eval_cases()

        tools = [args.skill] if args.skill else sorted(VALID_TOOLS)

        reports = [diagnose_skill(entries, eval_cases, t) for t in tools]
        neg = diagnose_negatives(entries)
        duals = diagnose_duals(entries)

        if args.skill:
            print_diagnosis(reports[0])
        else:
            print_full_diagnosis(reports, neg, duals)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
