# Listen — Knowledge Index

Quick reference for the `listen` project — a live user state observability CLI tool.

## Architecture
- **Bun pipeline**: Transcribe → Skill Router → Execute Skills → Respond → Emit events
- **Swift menu bar app**: Moonshine-powered transcription with mic picker
- **Skill system**: Self-describing skills with registry + LLM router
- **Multi-tool classifier**: Unified Qwen3.5-2B with LoRA, 98.4% accuracy (63/64)
- **Intent vector**: 6-dimension extensible system with exponential decay + activation gate
- **Dashboard**: SSE-driven with radar chart, sparklines, decision log, gate indicator

## Key Files
- `src/listen/intent-vector.ts` — `DIMENSION_DEFS` config registry, `IntentVectorStore`, `ActivationGate`
- `src/listen/index.ts` — Main pipeline orchestration
- `src/listen/dashboard.ts` — Web dashboard with dynamic dimension discovery
- `experts/` — MLX fine-tuning pipeline (Python/uv)
- `start.sh` — Single-command launcher

## Dimensions (6-axis intent vector)
| Dimension | Range | Source | Half-Life |
|-----------|-------|--------|-----------|
| accommodator | [0,1] | classifier | 45s |
| wellbeing | [0,1] | classifier | 120s |
| engagement | [0,1] | computed (chunk density) | 60s |
| taskFocus | [0,1] | computed (match ratio) | 30s |
| mood | [-1,1] | computed (sentiment) | 120s |
| energy | [0,1] | computed (speech rate) | 60s |

## Research
- `.knowledge/intent-vectors.md` — Continuous intent tracking approaches (EMA embeddings, structured vectors, DST, steering vectors, online learning)
- `.knowledge/datasets.md` — Public datasets for Phase 3 training (70+ datasets across mood, wellbeing, energy, engagement, intent classification)
- `.knowledge/competitive-analysis.md` — External tool landscape notes (Parakeet, RCLI) and adoption recommendations

## Search Keywords
`mlx`, `qwen`, `lora`, `fine-tuning`, `skill router`, `moonshine`, `transcription`,
`accommodator skill`, `wellbeing skill`, `afplay`, `elevenlabs`, `eval`, `expert models`,
`intent vector`, `intent tracking`, `user state`, `steering vectors`, `dialogue state`,
`dimension`, `mood`, `energy`, `engagement`, `activation gate`, `hysteresis`,
`CLINC150`, `MASSIVE`, `GoEmotions`, `EmoBank`, `NRC VAD`, `sentiment`, `arousal`
