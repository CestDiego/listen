"""Shared configuration for skill expert models."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SKILLS_DIR = ROOT / "skills"

# ── Base model ─────────────────────────────────────────────────────
# Qwen3.5-2B — step up from 0.8B for better multi-tool calling.
# Same hybrid DeltaNet architecture, BFCL-V4 score 43.6 (beats Qwen3-4B
# on agent/tool tasks). Native <tool_call> tokens, Apache 2.0.
# Memory: ~2.5GB in 8-bit, fits comfortably on Apple Silicon.
BASE_MODEL = "Qwen/Qwen3.5-2B"

# ── Training defaults ──────────────────────────────────────────────
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_LAYERS = 8        # Fine-tune last N layers
DEFAULT_BATCH_SIZE = 2         # Small for M-series memory
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_ITERS = 200            # Starting point — autoresearch may adjust

# ── Skills ─────────────────────────────────────────────────────────
# Each skill gets its own binary expert: "does this transcript match me?"
SKILLS = {
    "wellbeing": {
        "actions": ["check_in"],
        "description": "Negative self-talk, burnout, self-doubt, imposter syndrome",
    },
    "accommodator": {
        "actions": ["activate", "deactivate", "skip", "set_target"],
        "description": "Mood-adaptive music and environment adjustment",
    },
}

# ── Multi-tool definitions ─────────────────────────────────────────
# Used by the unified multi-tool model. Qwen3.5 native tool format.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "wellbeing.check_in",
            "description": "Triggered when the user expresses first-person negative self-talk, burnout, self-doubt, imposter syndrome, or emotional distress. NOT for third-person reports or casual tiredness.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accommodator.activate",
            "description": "User wants mood-adaptive music or environment adjustment. Covers requests to play music, start listening, or engage the mood-responsive system.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accommodator.skip",
            "description": "User wants to skip the current track without stopping or restarting the mood system. Covers: 'skip this song', 'next track', 'play the next one'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accommodator.deactivate",
            "description": "User wants to stop mood accommodation. Covers requests to stop, pause, or turn off the adaptive music/environment system.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accommodator.set_target",
            "description": "User explicitly states a desired mood or preference. Examples: 'I want to feel calm', 'help me focus', 'I like jazz', 'something upbeat please'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Pre-built tools JSON for reference (used by serve_multitool for /health endpoint)
def tools_json() -> str:
    """Return TOOL_DEFINITIONS as formatted JSON."""
    import json
    return json.dumps(TOOL_DEFINITIONS, indent=2)
