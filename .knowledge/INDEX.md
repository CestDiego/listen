# Listen — Knowledge Index

Quick reference for the `listen` project — a live user state observability CLI tool.

## Architecture
- **Bun pipeline**: Transcribe → Skill Router → Execute Skills → Respond → Emit events
- **Swift menu bar app**: Moonshine-powered transcription with mic picker
- **Skill system**: Self-describing skills with registry + LLM router
- **Expert models**: Per-skill fine-tuned Qwen3.5-0.8B via MLX LoRA (replaces 9B router)

## Key Files
- `src/listen/skills/router.ts` — LM Studio-based multi-skill router
- `experts/` — MLX fine-tuning pipeline (Python/uv)
- `start.sh` — Single-command launcher
- `.knowledge/` — This wiki

## Search Keywords
`mlx`, `qwen`, `lora`, `fine-tuning`, `skill router`, `moonshine`, `transcription`,
`music skill`, `wellbeing skill`, `apple music`, `elevenlabs`, `eval`, `expert models`
