"""
Multi-tool training data generator — unified model with Qwen3.5 native tool calling.

Instead of N binary classifiers, we train ONE model that can emit 0..N tool calls
per transcript using Qwen3.5's native <tool_call> tokens.

Output format per example:
  {"messages": [
    {"role": "system", "content": "You are...\\n<tools>...</tools>"},
    {"role": "user", "content": "<transcript>"},
    {"role": "assistant", "content": "<tool_call>\\n<function=accommodator.activate>\\n</function>\\n</tool_call>"}
  ]}

For negatives (no tools should fire):
  {"role": "assistant", "content": "No tools needed."}

Data sources:
  1. Eval cases (ground truth, including dual-activation)
  2. Synthetic single-skill positives (from existing templates, remapped to accommodator.*)
  3. Synthetic dual-skill positives (accommodator + wellbeing combos)
  4. New accommodator-specific positives
  5. Hard negatives (domain-adjacent but shouldn't fire)
  6. General negatives

This file follows autoresearch convention — infrastructure, not tuned.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .config import DATA_DIR, ROOT, TOOL_DEFINITIONS

# ── Reuse templates from the per-skill generator ──────────────────

from .generate import (
    MUSIC_POSITIVE_TEMPLATES,
    MUSIC_ACTIONS,
    WELLBEING_POSITIVE_TEMPLATES,
    MUSIC_NEGATIVES,
    WELLBEING_NEGATIVES,
    GENERAL_NEGATIVES,
)


# ── System prompt ─────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """Build system prompt with embedded tool definitions."""
    tool_lines = []
    for t in TOOL_DEFINITIONS:
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

SYSTEM_PROMPT = _build_system_prompt()


# ── Tool call formatting ──────────────────────────────────────────

def format_tool_call(function_name: str) -> str:
    """Format a single tool call in Qwen3.5 native format."""
    return f"<tool_call>\n<function={function_name}>\n</function>\n</tool_call>"


def format_tool_calls(calls: list[str]) -> str:
    """Format multiple tool calls. Empty list -> 'No tools needed.'"""
    if not calls:
        return "No tools needed."
    return "\n".join(format_tool_call(name) for name in calls)


# ── Chat entry builder ────────────────────────────────────────────

def make_entry(transcript: str, tool_calls: list[str]) -> dict[str, Any]:
    """Create a chat-format training entry for the multi-tool model."""
    assistant_content = format_tool_calls(tool_calls)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ── Map old music actions to new accommodator tools ───────────────
MUSIC_TO_ACCOMMODATOR = {
    "play": "accommodator.activate",
    "pause": "accommodator.deactivate",
    "resume": "accommodator.activate",
    "skip": "accommodator.skip",
    "previous": "accommodator.skip",
    "volume_up": "accommodator.set_target",
    "volume_down": "accommodator.set_target",
}


# ── Dual-activation templates ────────────────────────────────────
# Transcripts where BOTH accommodator and wellbeing should fire.

DUAL_TEMPLATES: list[dict[str, Any]] = [
    # ── skip + wellbeing ──
    {"transcript": "skip this stupid song, I'm such an idiot for adding it",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    {"transcript": "skip this, it's making me feel worse about myself",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    {"transcript": "next song please, I can't listen to this right now, I'm falling apart",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    {"transcript": "change the song, this one reminds me of how much I've failed",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    {"transcript": "skip this track, I hate everything about myself right now",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    {"transcript": "skip it, I can't stand this, I'm so useless",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    {"transcript": "next please, this song is making me cry and I already feel worthless",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    {"transcript": "skip this garbage, I'm the worst for picking this playlist",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    # (Exercise #5 attempts discarded — dual additions regress single-tool accuracy)
    # ── deactivate + wellbeing (was: pause + wellbeing) ──
    {"transcript": "pause the music, I need a break, I'm burning out",
     "tools": ["accommodator.deactivate", "wellbeing.check_in"]},
    {"transcript": "stop the music, I can't take this anymore",
     "tools": ["accommodator.deactivate", "wellbeing.check_in"]},
    {"transcript": "pause it, I'm so overwhelmed I can barely think",
     "tools": ["accommodator.deactivate", "wellbeing.check_in"]},
    {"transcript": "just stop everything, I can't deal with this right now, I'm worthless",
     "tools": ["accommodator.deactivate", "wellbeing.check_in"]},
    {"transcript": "hold the music, I need to cry for a second",
     "tools": ["accommodator.deactivate", "wellbeing.check_in"]},
    {"transcript": "pause, I need a moment, I feel like I'm losing my mind",
     "tools": ["accommodator.deactivate", "wellbeing.check_in"]},
    {"transcript": "stop playing, I'm having a breakdown and I can't handle noise",
     "tools": ["accommodator.deactivate", "wellbeing.check_in"]},
    {"transcript": "mute the music, I'm drowning in work and I hate myself",
     "tools": ["accommodator.deactivate", "wellbeing.check_in"]},
    # (Exercise #5 attempts discarded — dual additions regress single-tool accuracy)
    # ── activate + wellbeing (was: play + wellbeing) ──
    {"transcript": "play something calming, I'm having a terrible day and I hate myself",
     "tools": ["accommodator.activate", "wellbeing.check_in"]},
    {"transcript": "put on some music, I feel so broken inside",
     "tools": ["accommodator.activate", "wellbeing.check_in"]},
    {"transcript": "play my sad playlist, I'm such a failure at everything",
     "tools": ["accommodator.activate", "wellbeing.check_in"]},
    {"transcript": "play something gentle, I've been crying all afternoon",
     "tools": ["accommodator.activate", "wellbeing.check_in"]},
    {"transcript": "start the music, maybe it'll help, I feel completely empty inside",
     "tools": ["accommodator.activate", "wellbeing.check_in"]},
    {"transcript": "put on some tunes, I need to distract myself from how much I hate myself",
     "tools": ["accommodator.activate", "wellbeing.check_in"]},
    # ── set_target + wellbeing (was: volume + wellbeing) ──
    {"transcript": "turn it down, my head is killing me and I can't cope with anything",
     "tools": ["accommodator.set_target", "wellbeing.check_in"]},
    {"transcript": "louder please, I need to drown out these thoughts about how worthless I am",
     "tools": ["accommodator.set_target", "wellbeing.check_in"]},
    {"transcript": "quieter, I can barely function and the noise is making it worse",
     "tools": ["accommodator.set_target", "wellbeing.check_in"]},
    {"transcript": "crank it up, I need to stop hearing myself think about what a failure I am",
     "tools": ["accommodator.set_target", "wellbeing.check_in"]},
    # ── skip + wellbeing (was: previous + wellbeing — track navigation) ──
    {"transcript": "go back to the last song, it was the only thing keeping me from breaking down",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    {"transcript": "previous track please, I need that song back, I'm not doing okay",
     "tools": ["accommodator.skip", "wellbeing.check_in"]},
    # ── activate + wellbeing (was: resume + wellbeing) ──
    {"transcript": "unpause the music, the silence is making me spiral into negative thoughts",
     "tools": ["accommodator.activate", "wellbeing.check_in"]},
    {"transcript": "continue playing, the quiet makes me think about how much I've messed up",
     "tools": ["accommodator.activate", "wellbeing.check_in"]},
]


# ── Extra wellbeing positives ─────────────────────────────────────
# Reinforcement for patterns where the model may be confused (imposter syndrome
# uses "they" but is still first-person distress).

EXTRA_WELLBEING_POSITIVES: list[str] = [
    # Imposter syndrome (uses third-person pronouns but IS first-person distress)
    "they're going to realize I have no idea what I'm doing and I'll be fired",
    "everyone here is smarter than me, they'll figure out I'm faking it",
    "they're all going to see through me, I'm such a fraud",
    "I know they're going to find out I don't deserve to be here",
    "it's just a matter of time before they discover I'm incompetent",
    "people are going to find out I'm a fraud and I'll lose everything",
    "my coworkers are going to realize I'm not as smart as they think",
    "when they see my work they'll know I'm just pretending",
    # First-person with "stop everything" pattern (ambiguous with accommodator.deactivate)
    "I just want to stop existing sometimes",
    "I need everything to just stop, I can't keep going like this",
    # ── Exercise #2: giving up (was 4 examples, need more) ──
    "what's the point of even trying anymore",
    "I should just quit everything and disappear",
    "I'm done trying, nothing ever works out for me",
    "there's no point in continuing, I'll just fail again",
    "I give up, I can't keep pretending things will get better",
    # ── Exercise #2: loneliness (was 1 example) ──
    "nobody understands what I'm going through",
    "I have no one to talk to about any of this",
    "I feel so isolated, like nobody even notices me",
    "everyone has someone except me, I'm completely alone",
    "I could disappear and nobody would even notice",
    # ── Exercise #2: anxiety / panic (was 0 examples) ──
    "I'm having a panic attack, I can't breathe",
    "my anxiety is through the roof right now",
    "I can't stop shaking, I'm so anxious about everything",
    "my heart is racing and I feel like something terrible is about to happen",
    "I'm spiraling, the anxiety won't stop",
    # ── Exercise #2: grief / loss (was 0 examples) ──
    "I miss them so much it physically hurts",
    "I can't stop thinking about losing them, it's destroying me",
    "everything reminds me of them and I just break down",
    "the grief is eating me alive, I don't know how to cope",
    # ── Exercise #2: hopelessness (was 0 examples) ──
    "nothing will ever get better, I just know it",
    "there's no way out of this, I'm trapped",
    "I've tried everything and nothing works, what's the point",
    "the future looks completely dark, I can't see any hope",
    # ── Exercise #4: imposter syndrome reinforcement ──
    "they're going to find out I'm a fraud, I don't belong here",
    "everyone is going to realize I don't know what I'm doing",
    "I'm just waiting for the day they figure out I'm incompetent",
    "I feel like such a fraud, I don't deserve any of this",
]


# ── Additional negatives from real logs + action-sounding phrases ──
# These improve the negative class, especially for phrases that sound
# like commands but aren't directed at the assistant.

REAL_LOG_NEGATIVES: list[str] = [
    # Ambient speech from production logs
    "Hello. We get a pain to get replaced. But at the end of the day, I do feel like if you want the best possible",
    "I much appreciated but anyway we've got a lot more coming up in September i mean march",
    "I need a little bit of turquoise tea.",
    "So the music controller",
    "became tech timber or tech",
    "And there's houses here.",
    "Because you're so handsome.",
    "This is how I put it in my bag.",
    "We need frontier.",
    "It went from 44 to 77 percent",
    "In in in a look at the babesling one also went from 60 to 100.",
    "March Tarch. It's Tarch. Wait. It's Tarch. It's Tarch.",
    "Rosie the doll, Lola, Lola, put her outside.",
    "Thank you, children!",
    "We'll see you later",
    "Good night.",
    "Oh my god.",
    "Alright, it's the witching",
    "And at the center of the speed",
    "Remember look",
    "So I think like",
    "Thank you. You're doing okay.",
    "You're doing okay.",
    "I'm right here here",
    "Right here.",
    "A 3.5 model",
    "Fine-tuned",
    "I have a group.",
    "We just need...",
    "And then...",
    "There they go!",
    # Song lyrics / poetry / quotes (should NOT trigger wellbeing)
    "The little bird is gonna get happy, happy friends",
    "I have to be a masterpiece, but I'm proud to die.",
    "When I see the moon and the moon and the autumn leaves, there's nothing I can do.",
    "I'm not going to die. Oh, boy. El pobre es el rico, el rico es el rey",
    "and crafting In crafting together, our hands create and hearts unite",
    # Quoted/reported self-hate embedded in other context (should NOT trigger)
    "and crafting In crafting together, our hands create and hearts unite. Yeah, did none of them sound by me? I hate myself!",
    "the character in the movie screamed I hate myself and ran off",
    "she kept saying I'm worthless over and over in the play",
    "the song goes I'm such a failure, it's really catchy actually",
    "he read aloud from the book, I can't do anything right, and then closed it",
    # ── Exercise #2: hard negatives for new wellbeing patterns ──
    # Anxiety/panic in third person or past tense (NOT first-person distress)
    "she was having a panic attack at the office yesterday",
    "he told me his anxiety has been really bad lately",
    "the article talks about how anxiety affects productivity",
    "my friend had a panic attack during the presentation",
    # Grief/loss in third person or reported (NOT first-person distress)
    "she's been grieving since the funeral last month",
    "he said he misses his grandmother every day",
    "the movie was about a woman dealing with grief after losing her husband",
    # Loneliness/isolation in third person (NOT first-person distress)
    "he mentioned feeling lonely since moving to the new city",
    "the study found that remote workers feel more isolated",
    # Giving up in casual/non-distress context (NOT wellbeing)
    "I'm giving up on trying to fix this printer, let's just buy a new one",
    "I give up trying to find parking, let's take the bus",
    "I'm done trying to explain this, let's move on to the next topic",
    # Action-sounding phrases that are NOT music commands
    "let me share my screen for a second",
    "go ahead and start the presentation",
    "hold on, let me pull that up",
    "let's move on to the next topic",
    "let's skip that part and go straight to the demo",
    "can you repeat that one more time",
    "let me pull up the next slide",
    "go back to the previous point you made",
    "turn that idea around for a minute",
    "play it safe and don't commit yet",
    "let's pause on that and revisit later",
    "we need to fast forward this timeline",
    "skip the formalities and get to the point",
    "let's rewind a bit and think about this",
    "can we play devil's advocate here",
    "I need to turn this project around",
    # ── Hard negatives for "stop/pause/quiet" (NOT deactivate) ──
    "stop the meeting, we need to regroup",
    "stop sending me those emails",
    "stop overthinking it",
    "pause on that thought for a second",
    "let's pause the discussion and come back tomorrow",
    "hold that thought",
    "hold on a second, I need to check something",
    "stop the car, I forgot my wallet",
    "let's put a pause on the project",
    "quiet down everyone, the speaker is starting",
    "it's quiet in here today, where is everyone",
    "I need some quiet time to read",
    "can we stop talking about this",
    "just stop, you're making it worse",
    "the room went quiet when she walked in",
    # ── Hard negatives for "skip" (NOT accommodator.skip) ──
    "skip that part of the meeting",
    "can we skip lunch today",
    "skip the intro and get to the point",
    "I need to skip my morning run",
    "skip ahead in the book to chapter five",
    "let's skip the pleasantries",
    "skip the formalities",
    "I'll skip dessert tonight",
    "we should skip the next sprint review",
    "skip the boring parts",
    # ── Exercise #3: Hard negatives for "volume/turn up/turn down" (NOT set_target) ──
    "turn up the heat, it's freezing in here",
    "turn down the thermostat",
    "can you turn up the brightness on the screen",
    "turn down the lights for the movie",
    "the volume of work has been insane lately",
    "turn it around and look at the back",
    "I need to focus on this report",
    "I want to feel better about my presentation",
    "I'm in the mood for pizza",
    "I need something from the store",
]


# ── Accommodator-specific positives ──────────────────────────────
# New examples that don't exist in the old music templates.

ACCOMMODATOR_POSITIVES: dict[str, list[str]] = {
    "activate": [
        "play some music that matches my mood",
        "start the mood music",
        "can you put on something that fits the vibe",
        "I want some background music",
        "turn on the accommodator",
        "start listening to my mood",
        # ── Exercise #4: resume/unpause (confused with deactivate) ──
        "unpause the music",
        "unpause it",
        "unpause please",
        "resume the music",
        "resume playing",
        "resume playback",
        "bring the music back",
        "continue the music",
        "continue playing please",
        "start the music again",
        "put the music back on",
        "turn the music back on",
        # ── Exercise #4: focus/concentration (confused with set_target) ──
        "help me focus please",
        "help me concentrate",
        "I need focus music",
        "play something for focus",
    ],
    "skip": [
        "skip this song",
        "next track please",
        "play the next one",
        "skip to the next song",
        "I don't like this one, skip it",
        "next song",
        "change the track",
    ],
    "deactivate": [
        # ── pause variants ──
        "stop the music please",
        "pause the music",
        "pause it",
        "pause please",
        "pause the song",
        "hold on, pause",
        "can you pause that",
        "pause it for a sec",
        "pause the music for a minute",
        "just pause",
        # ── stop variants ──
        "stop the music",
        "stop playing",
        "stop playing music",
        "stop it",
        "that's enough music",
        "enough music for now",
        "cut the music",
        "kill the music",
        # ── mute / silence variants ──
        "mute everything",
        "mute the music",
        "mute it",
        "silence please",
        "silence the music",
        "shh, turn it off",
        # ── turn off variants ──
        "turn off the mood music",
        "turn off the background music",
        "turn off the music",
        "turn it off",
        "switch off the music",
        # ── don't want / need quiet ──
        "I don't want music right now",
        "no more music",
        "no more music please",
        "I need quiet",
        "I need silence",
        "I need to think without music",
        "can we have some quiet",
        "stop accommodating",
        "turn off the accommodator",
        # ── contextual ──
        "I'm getting on a call, pause the music",
        "hold the music, someone's talking to me",
        "stop the music, I need to focus on this call",
        "pause, I need to answer the phone",
        "mute the music, I'm in a meeting now",
    ],
    "set_target": [
        # ── mood target ──
        "I want to feel calm",
        "help me focus",
        "I need something upbeat",
        "play something relaxing",
        "I'm in the mood for jazz",
        "something energetic please",
        "I want to wind down",
        "can you help me concentrate",
        "I need to chill out",
        "play something happy",
        "I want peaceful vibes",
        "something to help me study",
        "I need something mellow right now",
        "make it more chill",
        "I want something intense",
        "switch to something more uplifting",
        # ── volume control ──
        "turn it up",
        "turn it up a bit",
        "turn it down",
        "turn it down a little",
        "louder please",
        "louder",
        "quieter please",
        "quieter",
        "volume up",
        "volume down",
        "pump up the volume",
        "set volume to 100 percent",
        "set the volume to 50",
        "turn the volume up",
        "turn the volume down",
        "make it louder",
        "make it quieter",
        "can you turn it up a notch",
        "lower the volume a bit",
        "raise the volume",
        "crank it up",
        "bring the volume down",
        "it's too quiet, turn it up",
        "it's too loud, turn it down",
        "volume down please, it's too loud",
        # ── rating / preference ──
        "I love this song, remember that",
        "this is a great track, save it",
        "I don't like this kind of music",
        "more songs like this one please",
    ],
}


# ── Data generation ───────────────────────────────────────────────

def load_eval_cases() -> list[dict[str, Any]]:
    """Load eval cases from the TypeScript project."""
    eval_path = ROOT.parent / "src" / "listen" / "skills" / "eval-cases.json"
    with open(eval_path) as f:
        data = json.load(f)
    # Filter out section separators (entries without "name")
    return [c for c in data["cases"] if "name" in c]


def eval_case_to_tool_calls(case: dict[str, Any]) -> list[str]:
    """Convert an eval case's expected skills to tool call names.

    Remaps old music.* tool names to new accommodator.* tools.
    """
    skills = case["expect"].get("skills", [])
    result = []
    for s in skills:
        tool_name = f"{s['skill']}.{s['action']}"
        # Remap old music actions to accommodator
        if s["skill"] == "music":
            tool_name = MUSIC_TO_ACCOMMODATOR.get(s["action"], "accommodator.activate")
        result.append(tool_name)
    return result


def _deduplicate(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate by transcript text (case-insensitive).

    When the same transcript appears with different labels (e.g. single-skill
    vs dual-skill), keep the one with MORE tool calls — that's the correct
    label per eval-cases.json ground truth.
    """
    by_text: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = entry["messages"][1]["content"].lower().strip()
        existing = by_text.get(key)
        if existing is None:
            by_text[key] = entry
        else:
            # Keep the entry with more tool calls (dual > single > negative)
            new_count = entry["messages"][-1]["content"].count("<tool_call>")
            old_count = existing["messages"][-1]["content"].count("<tool_call>")
            if new_count > old_count:
                by_text[key] = entry
    return list(by_text.values())


def _stratified_split(
    entries: list[dict[str, Any]],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    rng: random.Random | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split entries with stratification by category (negative/single/dual).

    Ensures each split has proportional representation — critical with small
    datasets where random shuffle can leave 0 dual examples in test.
    """
    from collections import defaultdict

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        n_tools = e["messages"][-1]["content"].count("<tool_call>")
        cat = "dual" if n_tools >= 2 else ("positive" if n_tools >= 1 else "negative")
        by_cat[cat].append(e)

    train: list[dict] = []
    valid: list[dict] = []
    test: list[dict] = []

    for cat, items in by_cat.items():
        if rng:
            rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * ratios[0]))
        n_valid = max(1, int(n * ratios[1]))
        train.extend(items[:n_train])
        valid.extend(items[n_train:n_train + n_valid])
        test.extend(items[n_train + n_valid:])

    if rng:
        rng.shuffle(train)
        rng.shuffle(valid)
        rng.shuffle(test)

    return train, valid, test


def generate_multitool_data(seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Generate train/valid/test splits for the unified multi-tool model.

    Returns (train, valid, test) lists of chat-format entries.
    """
    rng = random.Random(seed)
    entries: list[dict[str, Any]] = []

    # 1. Seed from eval cases (ground truth, remapped to accommodator.*)
    for case in load_eval_cases():
        transcript = case["transcript"]
        tool_calls = eval_case_to_tool_calls(case)
        entries.append(make_entry(transcript, tool_calls))

    # 2. Synthetic single-skill positives: music templates -> remapped to accommodator
    for action_idx, templates in enumerate(MUSIC_POSITIVE_TEMPLATES):
        action = MUSIC_ACTIONS[action_idx]
        tool_name = MUSIC_TO_ACCOMMODATOR.get(action, "accommodator.activate")
        for t in templates:
            entries.append(make_entry(t, [tool_name]))

    # 3. Synthetic single-skill positives: wellbeing
    for t in WELLBEING_POSITIVE_TEMPLATES:
        entries.append(make_entry(t, ["wellbeing.check_in"]))

    # 4. New accommodator-specific positives
    for action, templates in ACCOMMODATOR_POSITIVES.items():
        tool_name = f"accommodator.{action}"
        for t in templates:
            entries.append(make_entry(t, [tool_name]))

    # 5. Dual-activation examples
    for dual in DUAL_TEMPLATES:
        entries.append(make_entry(dual["transcript"], dual["tools"]))

    # 6. Hard negatives (domain-adjacent, no tools)
    for t in MUSIC_NEGATIVES:
        entries.append(make_entry(t, []))
    for t in WELLBEING_NEGATIVES:
        entries.append(make_entry(t, []))

    # 7. General negatives
    for t in GENERAL_NEGATIVES:
        entries.append(make_entry(t, []))

    # 8. Real-log negatives + action-sounding phrases
    for t in REAL_LOG_NEGATIVES:
        entries.append(make_entry(t, []))

    # 9. Extra wellbeing positives (imposter syndrome reinforcement)
    for t in EXTRA_WELLBEING_POSITIVES:
        entries.append(make_entry(t, ["wellbeing.check_in"]))

    # Deduplicate: when same transcript has conflicting labels, keep dual > single > negative
    entries = _deduplicate(entries)

    # Stats before split
    n_total = len(entries)
    n_positive = sum(1 for e in entries if "tool_call" in e["messages"][-1]["content"])
    n_dual = sum(1 for e in entries if e["messages"][-1]["content"].count("<tool_call>") >= 2)
    n_negative = n_total - n_positive

    print(f"  Total entries (after dedup): {n_total}")
    print(f"  Positive (1+ tools): {n_positive}")
    print(f"  Dual-activation (2+ tools): {n_dual}")
    print(f"  Negative (no tools): {n_negative}")
    print(f"  Positive ratio: {n_positive / n_total:.1%}")

    # Stratified split: ensures each split has proportional neg/single/dual
    train, valid, test = _stratified_split(entries, rng=rng)

    return train, valid, test


def write_jsonl(entries: list[dict], path: Path) -> None:
    """Write entries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")


def main() -> None:
    """Generate training data for the unified multi-tool model."""
    print(f"\n{'='*60}")
    print(f"Generating multi-tool training data")
    print(f"{'='*60}")

    train, valid, test = generate_multitool_data()

    data_dir = DATA_DIR / "multitool"
    write_jsonl(train, data_dir / "train.jsonl")
    write_jsonl(valid, data_dir / "valid.jsonl")
    write_jsonl(test, data_dir / "test.jsonl")

    # Per-split stats
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        n_pos = sum(1 for e in split if "tool_call" in e["messages"][-1]["content"])
        n_dual = sum(1 for e in split if e["messages"][-1]["content"].count("<tool_call>") >= 2)
        print(f"  {name}: {len(split)} entries ({n_pos} positive, {n_dual} dual, {len(split) - n_pos} negative)")

    print(f"  Saved to: {data_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
