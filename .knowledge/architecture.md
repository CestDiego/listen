# Listen — Architecture

## Multi-Tool Classifier (MLX)

### Overview
A single unified Qwen3.5-2B model with LoRA adapter classifies transcripts into
zero or more tool calls using native `<tool_call>` tokens. Replaces the previous
per-skill binary expert architecture (and the original 9B LM Studio router before that).

### Base Model
**Qwen3.5-2B** (Apache 2.0)
- Hybrid architecture: Gated DeltaNet + Gated Attention
- 1.9B params, ~2.5GB in 8-bit on Apple Silicon
- BFCL-V4 score 43.6 — beats Qwen3-4B on agent/tool tasks
- Native tool calling tokens: `<tool_call>` (248058), `</tool_call>` (248059)
- 262K context window

### Tool Definitions
```
music.play       — Start playing music or a specific song/playlist
music.pause      — Pause currently playing music
music.resume     — Resume paused music
music.skip       — Skip to next track
music.previous   — Go back to previous track
music.volume_up  — Increase volume
music.volume_down — Decrease volume
wellbeing.check_in — First-person negative self-talk, burnout, self-doubt, imposter syndrome
```

### Training Pipeline (autoresearch-inspired)
```
experts/
├── experts/
│   ├── config.py                      # Shared config (base model, tools, defaults)
│   ├── generate.py                    # READ-ONLY: per-skill data gen (templates)
│   ├── evaluate.py                    # READ-ONLY: per-skill eval
│   ├── train.py                       # Per-skill training (legacy fallback)
│   ├── serve.py                       # Per-skill worker pool server (legacy fallback)
│   ├── generate_multitool.py          # Multi-tool data gen with dedup + stratified split
│   ├── evaluate_multitool.py          # Multi-tool eval with VALID_TOOLS allowlist parser
│   ├── evaluate_multitool_integration.py  # 64-case integration eval
│   ├── train_multitool.py             # Unified LoRA fine-tuning (rank 16, 12 layers)
│   └── serve_multitool.py             # Single-model HTTP server, backward-compatible API
├── data/
│   └── multitool/                     # train.jsonl, valid.jsonl, test.jsonl (291 entries)
├── models/
│   └── multitool/                     # adapters.safetensors/, train_meta.json, eval_test.json
└── program.md                         # Human-editable methodology spec
```

### Training Data (Round 4)
- 291 entries after dedup, 57.4% positive, 30 dual-activation, 124 negative
- Stratified split: train 202, valid 42, test 47
- Sources: eval cases (64), music templates (55), wellbeing templates (46),
  dual templates (30), hard negatives (20+20), general negatives (20),
  real-log negatives (56), extra wellbeing (10)

### Training Hyperparameters
- LoRA rank: 16, layers: 12, LR: 5e-5, batch size: 2, iterations: 300
- `--mask-prompt`: only trains on response (tool call output), not system prompt
- Peak memory: 14.6GB, training time: ~8 minutes on Apple Silicon

### Results — Evolution

| Round | Model | Changes | Accuracy | Dual | Negatives | Avg Latency |
|-------|-------|---------|----------|------|-----------|-------------|
| 1 | 0.8B | Initial (22 cases) | 22/22 (100%) | 2/2 | 7/7 | 252ms |
| 2 | 0.8B | +negatives, +dedup, +stratified (64 cases) | 59/64 (92.2%) | 0/4 | 34/35 | 234ms |
| 3 | 0.8B | +more dual examples (15→30) | 61/64 (95.3%) | 3/4 | 33/35 | 243ms |
| 4 | 0.8B | +imposter reinforcement | 60/64 (93.8%) | 3/4 | 33/35 | 239ms |
| **5** | **2B** | **Model upgrade to Qwen3.5-2B** | **63/64 (98.4%)** | **4/4** | **34/35** | **368ms** |

Round 1 was on the original 22-case set; rounds 2-5 on the expanded 64-case set.

### Architecture Diagram
```
OLD (single router, deprecated):
  Transcript → [9B LM Studio] → {skills: [{skill: "music", action: "play"}]}
  ~1,400ms per call, 86% accuracy

PREVIOUS (parallel per-skill experts):
  Bun → POST /v1/classify → HTTP server → dispatch to worker pool
                             ├─ [Worker 1: music LoRA]    → {match, action}
                             └─ [Worker 2: wellbeing LoRA] → {match, action}
                             ← merge results
  ~286ms avg, 95.5% accuracy (21/22)

CURRENT (unified multi-tool):
  Bun → POST /v1/classify → serve_multitool.py → single Qwen3.5-2B + LoRA
                             → parse <tool_call> blocks → VALID_TOOLS filter
                             ← {skills: [{skill, action, confidence}]}
  ~368ms avg, 98.4% accuracy (63/64), supports dual-activation
```

### Key Learnings

1. **Single unified model beats per-skill experts** — one 2B model with tool definitions
   in the system prompt achieves 98.4% on a much harder 64-case eval set vs 95.5% on
   the original 22-case set with per-skill experts.

2. **Dual-activation works** — the model can emit multiple `<tool_call>` blocks in one
   response. 4/4 dual cases pass at 2B (was 3/4 at 0.8B, 0/4 initially).

3. **Tool definitions in system prompt are critical** — without them, the model
   hallucinated actions like `music.dislike`. Adding explicit tool list + VALID_TOOLS
   allowlist in the parser fixed this.

4. **Training data deduplication prevents contradictory labels** — same transcript
   appeared as both single-skill and dual-skill. Dedup keeps the dual version.

5. **Stratified splitting prevents empty categories** — random shuffle with 291 examples
   left 0 dual cases in test split. Stratified split ensures proportional representation.

6. **Real production logs are gold for eval** — mined 33 cases from
   `/tmp/listen-events.jsonl` and `/tmp/listen-session-*.json`, including false positives
   (song lyrics triggering wellbeing), ASR artifacts, ambient speech.

7. **Class imbalance kills recall** — First wellbeing attempt (37% positive) gave 0%
   recall. Expanding to 42% positive + lower LR (5e-5 vs 1e-4) fixed it completely.

8. **mlx-lm API changes** — `temp` parameter removed from `generate()`.
   Use `sampler=` with `mx.argmax` for greedy decoding instead.

9. **Adapter path is a directory** — `--adapter-path` in mlx-lm creates a directory
   containing `adapters.safetensors` + checkpoints.

10. **MLX parallel inference** — Python threading crashes MLX (shared Metal command
    buffer). Separate processes work but aren't needed with the unified model.

### Remaining Failure (Round 5)
- "not good enough" (real log watchlist) — model correctly fires `wellbeing.check_in`
  but also spuriously adds `music.skip`. Over-activation, not a miss.
- Fix: add negative reinforcement examples where "not good enough" appears without
  music context.

### Backward Compatibility
- The old per-skill pipeline (generate.py, train.py, evaluate.py, serve.py) still works
  as a fallback. Per-skill adapters remain in `models/music/` and `models/wellbeing/`.
- `serve_multitool.py` exposes the same `/v1/classify` API — the Bun client
  (`classify.ts`) needs zero code changes.
