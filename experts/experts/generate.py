"""
Training data generator — synthetic augmentation from eval cases.

Each skill expert is a binary classifier with action selection:
  Input:  transcript text
  Output: {"match": true, "action": "play", "confidence": 0.9}
       or {"match": false}

We generate training data in the chat format that mlx-lm expects:
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Data sources:
  1. Seed cases from eval-cases.json (ground truth)
  2. Synthetic augmentation via paraphrasing templates
  3. Hard negatives (transcripts that mention the skill domain but shouldn't fire)

This file is READ-ONLY per autoresearch convention — it's infrastructure.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .config import DATA_DIR, SKILLS, ROOT


# ── System prompts per skill ───────────────────────────────────────

def system_prompt(skill_name: str) -> str:
    """Build the system prompt for a skill expert."""
    skill = SKILLS[skill_name]
    actions_str = ", ".join(f'"{a}"' for a in skill["actions"])
    return (
        f"You are a binary classifier for the '{skill_name}' skill in a voice assistant.\n"
        f"Skill: {skill['description']}\n"
        f"Available actions: [{actions_str}]\n\n"
        f"Given a transcript, decide if this skill should activate.\n"
        f"Rules:\n"
        f"- Only activate for CLEAR intent or signal directed at the assistant.\n"
        f"- Do NOT activate when someone is merely talking ABOUT the topic.\n"
        f"- For wellbeing: ONLY activate for first-person self-directed distress, not third-person reports.\n"
        f'- Respond with ONLY a JSON object: {{"match": true, "action": "<action>", "confidence": <0.0-1.0>}}\n'
        f'  or: {{"match": false}}'
    )


# ── Paraphrase templates ──────────────────────────────────────────

MUSIC_POSITIVE_TEMPLATES: list[list[str]] = [
    # play
    ["put on some tunes", "start the music", "can you play something", "I want to hear music",
     "play me a song", "let's get some music going", "hit play", "queue up some music",
     "play my playlist", "start playing something nice", "throw on some beats"],
    # pause
    ["stop the music", "hold the music", "can you pause that", "mute the music",
     "I need quiet", "silence the music", "stop playing",
     "pause the music, I need a break, I'm burning out",
     "just stop everything, I can't deal with this right now"],
    # resume
    ["continue the music", "unpause", "keep playing", "bring the music back",
     "start it again", "continue where we left off"],
    # skip
    ["next song", "skip this one", "I don't like this track", "play the next one",
     "change the song", "skip ahead", "next please", "go to the next track",
     "skip this stupid song, I'm such an idiot for adding it",
     "skip this, it's making me feel worse",
     "just skip it, I can't listen to this right now"],
    # previous
    ["go back one song", "previous song", "play the last track again",
     "I liked the previous one", "back one track", "replay the last song"],
    # volume_up
    ["louder please", "crank it up", "turn the volume up", "make it louder",
     "I can barely hear it", "boost the volume", "pump up the volume"],
    # volume_down
    ["quieter please", "turn it down", "lower the volume", "too loud",
     "bring the volume down", "softer please", "it's way too loud"],
]

MUSIC_ACTIONS = ["play", "pause", "resume", "skip", "previous", "volume_up", "volume_down"]

WELLBEING_POSITIVE_TEMPLATES: list[str] = [
    # Self-criticism / worthlessness
    "I'm such a failure at everything",
    "I can't do anything right, I'm useless",
    "I hate myself for being so weak",
    "what's the point, I'll never be good enough",
    "I feel like a complete impostor here",
    "everyone would be better off without me",
    "I'm falling apart and I can't stop it",
    "I don't deserve any of this, I'm a fraud",
    "I'm so overwhelmed I can barely function",
    "I keep disappointing everyone around me",
    "I'm not smart enough for this job",
    "I feel broken inside",
    "I'm going to mess everything up like I always do",
    "I'm nothing compared to my colleagues",
    "I hate how weak I am",
    "I'll never measure up no matter how hard I try",
    "I feel completely alone in this",
    "I'm just pretending to know what I'm doing",
    "nobody actually cares about me",
    "I'm a terrible person for thinking this way",
    "I'm so stupid I can't believe I did that",
    "I'm worthless and nobody even cares",
    # Burnout
    "I'm drowning in work and nobody notices",
    "I just want to disappear sometimes",
    "why do I even bother trying anymore",
    "I'm so burned out I can barely think",
    "I feel like giving up on everything",
    "I can't take this anymore I'm completely exhausted",
    "I'm running on empty and nobody sees it",
    "I've been working nonstop and I feel like collapsing",
    "I don't have the energy to keep doing this",
    "every day feels the same and I'm losing it",
    # Imposter syndrome
    "they're going to find out I'm a fraud and fire me",
    "I don't belong here everyone else is so much smarter",
    "I got lucky and it's only a matter of time before they realize",
    "I'm in way over my head and pretending otherwise",
    "I feel like I'm faking it every single day",
    # Additional variations — ensure the model sees enough positive signal
    "I'm so dumb why did I say that",
    "I'm the worst at this, I should quit",
    "I messed up again, I always ruin everything",
    "I hate that I'm like this",
    "what's wrong with me, why can't I do anything right",
    "I feel so empty inside lately",
    "I've been crying at my desk and nobody knows",
    "I can't even look at myself in the mirror",
    "I'm losing control of everything",
    "I want to scream but I can't even do that",
]

# Hard negatives — things that MENTION the domain but shouldn't trigger
MUSIC_NEGATIVES: list[str] = [
    "I really liked that album we listened to yesterday",
    "the concert last week was amazing",
    "she plays guitar beautifully",
    "we should go to more music festivals",
    "that new song is stuck in my head all day",
    "the bass line in that track was incredible",
    "I used to play piano as a kid",
    "have you heard the new Taylor Swift album",
    "the DJ at the party was great",
    "my favorite genre is definitely jazz",
    "the soundtrack for that movie was perfect",
    "he's been taking drum lessons for a year now",
    "they're releasing a new single next week",
    "I saw them perform live twice this year",
    "that guitar riff is so iconic",
    "classical music helps me focus sometimes",
    "we talked about our favorite bands at lunch",
    "the music at that restaurant was too loud",
    "she recommended some great podcasts about music history",
    "the audio quality on those headphones is insane",
]

WELLBEING_NEGATIVES: list[str] = [
    "she said she was feeling really down lately",
    "he told me he hates his job",
    "my friend is going through a rough time",
    "the character in the movie was depressed",
    "I read an article about burnout in tech",
    "she mentioned she was exhausted from work",
    "he seems stressed but won't talk about it",
    "the book discusses imposter syndrome in detail",
    "she said she felt like giving up",
    "they were talking about mental health awareness",
    "I'm a bit tired today, had a late night",
    "the weather is making everyone feel gloomy",
    "yeah I stayed up late watching Netflix again",
    "I could use another coffee honestly",
    "this meeting is dragging on forever",
    "I'm hungry, when's lunch",
    "that was a tough workout this morning",
    "my kid had a rough day at school",
    "the news is really depressing these days",
    "I need to start getting more sleep",
]

# General negatives — no skill should fire
GENERAL_NEGATIVES: list[str] = [
    "let's schedule the standup for tomorrow at ten",
    "the weather is really nice today isn't it",
    "I need to pick up groceries after work",
    "can you send me that document by end of day",
    "the quarterly report looks good overall",
    "we should plan a team outing next month",
    "I just finished reading a great book",
    "traffic was terrible this morning",
    "what do you want for dinner tonight",
    "the new office layout is much better",
    "I have a dentist appointment on Thursday",
    "did you see the game last night",
    "we need to update the API documentation",
    "the deployment went smoothly this time",
    "I think we should refactor that module",
    "my flight is at six in the morning",
    "let me share my screen for a second",
    "the pull request needs one more review",
    "we're almost at the end of the sprint",
    "the coffee machine is broken again",
]


def make_chat_entry(skill_name: str, transcript: str, match: bool,
                    action: str = "", confidence: float = 0.9) -> dict[str, Any]:
    """Create a chat-format training entry."""
    if match:
        assistant_content = json.dumps(
            {"match": True, "action": action, "confidence": confidence},
            separators=(",", ":"),
        )
    else:
        assistant_content = json.dumps({"match": False}, separators=(",", ":"))

    return {
        "messages": [
            {"role": "system", "content": system_prompt(skill_name)},
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def load_eval_cases() -> list[dict[str, Any]]:
    """Load the eval cases from the TypeScript project."""
    eval_path = ROOT.parent / "src" / "listen" / "skills" / "eval-cases.json"
    with open(eval_path) as f:
        data = json.load(f)
    return data["cases"]


def generate_skill_data(skill_name: str, seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Generate train/valid/test splits for a skill.

    Returns (train, valid, test) lists of chat-format entries.
    """
    rng = random.Random(seed)
    entries: list[dict[str, Any]] = []

    # 1. Seed from eval cases
    for case in load_eval_cases():
        transcript = case["transcript"]
        expected_skills = case["expect"].get("skills", [])
        no_skills = case["expect"].get("noSkills", [])

        # Check if this skill should fire
        skill_match = next(
            (s for s in expected_skills if s["skill"] == skill_name),
            None,
        )

        if skill_match:
            entries.append(make_chat_entry(
                skill_name, transcript, True,
                action=skill_match["action"], confidence=0.95,
            ))
        elif skill_name in no_skills or not expected_skills:
            entries.append(make_chat_entry(skill_name, transcript, False))
        # If the case is for another skill but doesn't explicitly exclude us, skip

    # 2. Synthetic positives
    if skill_name == "music":
        for action_idx, templates in enumerate(MUSIC_POSITIVE_TEMPLATES):
            action = MUSIC_ACTIONS[action_idx]
            for t in templates:
                # Vary confidence slightly
                conf = round(rng.uniform(0.85, 0.98), 2)
                entries.append(make_chat_entry(skill_name, t, True, action=action, confidence=conf))

    elif skill_name == "wellbeing":
        for t in WELLBEING_POSITIVE_TEMPLATES:
            conf = round(rng.uniform(0.80, 0.95), 2)
            entries.append(make_chat_entry(skill_name, t, True, action="check_in", confidence=conf))

    # 3. Hard negatives (domain-adjacent but should NOT fire)
    negatives = (
        MUSIC_NEGATIVES if skill_name == "music" else WELLBEING_NEGATIVES
    )
    for t in negatives:
        entries.append(make_chat_entry(skill_name, t, False))

    # 4. General negatives
    for t in GENERAL_NEGATIVES:
        entries.append(make_chat_entry(skill_name, t, False))

    # 5. Cross-skill entries (other skill's positives are our negatives)
    if skill_name == "music":
        for t in WELLBEING_POSITIVE_TEMPLATES[:10]:
            entries.append(make_chat_entry(skill_name, t, False))
    elif skill_name == "wellbeing":
        for action_idx, templates in enumerate(MUSIC_POSITIVE_TEMPLATES):
            for t in templates[:3]:  # First 3 from each action
                entries.append(make_chat_entry(skill_name, t, False))

    # Shuffle and split: 70% train, 15% valid, 15% test
    rng.shuffle(entries)
    n = len(entries)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)

    train = entries[:n_train]
    valid = entries[n_train:n_train + n_valid]
    test = entries[n_train + n_valid:]

    return train, valid, test


def write_jsonl(entries: list[dict], path: Path) -> None:
    """Write entries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")


def main() -> None:
    """Generate training data for all skills."""
    for skill_name in SKILLS:
        print(f"\n{'='*60}")
        print(f"Generating data for: {skill_name}")
        print(f"{'='*60}")

        train, valid, test = generate_skill_data(skill_name)

        skill_dir = DATA_DIR / skill_name
        write_jsonl(train, skill_dir / "train.jsonl")
        write_jsonl(valid, skill_dir / "valid.jsonl")
        write_jsonl(test, skill_dir / "test.jsonl")

        # Stats
        pos_train = sum(1 for e in train if json.loads(e["messages"][-1]["content"]).get("match", False))
        pos_valid = sum(1 for e in valid if json.loads(e["messages"][-1]["content"]).get("match", False))
        pos_test = sum(1 for e in test if json.loads(e["messages"][-1]["content"]).get("match", False))

        print(f"  train: {len(train)} entries ({pos_train} positive, {len(train) - pos_train} negative)")
        print(f"  valid: {len(valid)} entries ({pos_valid} positive, {len(valid) - pos_valid} negative)")
        print(f"  test:  {len(test)} entries ({pos_test} positive, {len(test) - pos_test} negative)")
        print(f"  saved to: {skill_dir}/")

    print(f"\nDone. Total skills: {len(SKILLS)}")


if __name__ == "__main__":
    main()
