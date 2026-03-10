"""Shared configuration for skill expert models."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SKILLS_DIR = ROOT / "skills"

# ── Base model ─────────────────────────────────────────────────────
# Qwen3.5-0.8B — small enough for fast LoRA, big enough for classification.
# We use the instruct model (not base) since it already has a chat template
# trained for LoRA PEFT without needing to fine-tune embeddings.
BASE_MODEL = "Qwen/Qwen3.5-0.8B"

# ── Training defaults ──────────────────────────────────────────────
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_LAYERS = 8        # Fine-tune last N layers
DEFAULT_BATCH_SIZE = 2         # Small for M-series memory
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_ITERS = 200            # Starting point — autoresearch may adjust

# ── Skills ─────────────────────────────────────────────────────────
# Each skill gets its own binary expert: "does this transcript match me?"
SKILLS = {
    "music": {
        "actions": ["play", "pause", "resume", "skip", "previous", "volume_up", "volume_down"],
        "description": "Music playback control commands",
    },
    "wellbeing": {
        "actions": ["check_in"],
        "description": "Negative self-talk, burnout, self-doubt, imposter syndrome",
    },
}

# ── Multi-tool definitions ─────────────────────────────────────────
# Used by the unified multi-tool model. Qwen3.5 native tool format.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "music.play",
            "description": "Start playing music or play a specific song/playlist.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music.pause",
            "description": "Pause the currently playing music.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music.resume",
            "description": "Resume paused music.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music.skip",
            "description": "Skip to the next song/track.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music.previous",
            "description": "Go back to the previous song/track.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music.volume_up",
            "description": "Increase the music volume.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music.volume_down",
            "description": "Decrease the music volume.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wellbeing.check_in",
            "description": "Triggered when the user expresses first-person negative self-talk, burnout, self-doubt, imposter syndrome, or emotional distress. NOT for third-person reports or casual tiredness.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Pre-built tools JSON for reference (used by serve_multitool for /health endpoint)
def tools_json() -> str:
    """Return TOOL_DEFINITIONS as formatted JSON."""
    import json
    return json.dumps(TOOL_DEFINITIONS, indent=2)
