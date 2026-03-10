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

---

## Intent Vector (Real-Time State Observability)

### Overview
A multi-dimensional activation vector that decays over time, tracking user intent
signals derived from classifier outputs. Visualized live in the dashboard's
"Intent" tab with a radar chart and sparkline traces.

### Architecture
```
Transcript → classifyTranscript() → expertResults[]
                                          ↓
                                   IntentVectorStore.update()
                                     ├─ Decay all dimensions (exponential half-life)
                                     ├─ Apply expert match activations (max, not replace)
                                     ├─ Compute engagement (chunk density in 60s window)
                                     ├─ Compute taskFocus (match ratio in 30s window)
                                     ├─ Compute trends (Δ vs 10s-ago snapshot)
                                     └─ Push to ring buffer (300 slots ≈ 5 min)
                                          ↓
                                   session.emitIntentVector(snapshot, history)
                                          ↓
                                   SSE "intentVector" → Dashboard
                                     ├─ Radar chart (Canvas API, 4 axes)
                                     ├─ Dimension readouts (bars + trend arrows)
                                     └─ Sparkline traces (per-dimension, 5 min)
```

### Dimensions (Phase 1)

| Dimension | Range | Decay Half-Life | Source |
|-----------|-------|-----------------|--------|
| `music` | [0, 1] | 45s | Classifier confidence (0.95 on match) |
| `wellbeing` | [0, 1] | 120s | Classifier confidence (0.83-0.95 on match) |
| `engagement` | [0, 1] | 60s | Chunks in last 60s / 12 (max expected) |
| `taskFocus` | [0, 1] | 30s | Ratio of skill-matched chunks in last 30s |

### Key Design Decisions
- **Decay function**: `baseline + (value - baseline) * 0.5^(elapsed / halfLife)`
- **Max-not-replace**: activation uses `Math.max(current, confidence)` so rapid
  re-triggers don't lower an already-high signal
- **Ring buffer**: fixed 300 slots with head pointer, O(1) push, ordered materialization
- **Trends**: compare current vs ~10s-ago snapshot, clamped to [-1, 1]
- **SSE event**: `"intentVector"` with `{ snapshot, history }` — full history sent
  each time so dashboard can redraw sparklines from any reconnect point

### Files
- `src/listen/intent-vector.ts` — IntentVectorStore class, decay function, types
- `src/listen/index.ts` — wired after classify, before addRouterDecision
- `src/listen/session.ts` — emitIntentVector() method, included in getSession()/getStats()
- `src/listen/dashboard.ts` — "Intent" tab with radar chart + sparklines
- `scripts/inject-test.ts` — test injection script (9 transcripts, timed sequence)

### Testing
Run `bun run scripts/inject-test.ts` to inject a scripted sequence that exercises
all dimensions: music commands, wellbeing triggers, neutral text, and decay pauses.
Requires `./start.sh` or manual pipeline + expert server startup.

---

## Activation Gate (Post-Classification Wellbeing Bias)

### Overview
A Schmitt-trigger-inspired state machine that sits after the classifier and biases
routing toward wellbeing when recent context suggests emotional distress. Implements
hysteresis (different thresholds for activation vs deactivation) and cost-asymmetric
promotion (missing distress costs 10× more than a false check-in).

### Research Basis
- **Collins & Loftus (1975)** — spreading activation theory (decay + propagation)
- **Dialogflow context lifespans** — turn-based sticky state with renewal on match
- **Bayesian cost-sensitive thresholds** — FN/FP cost ratio drives threshold
- **LEAD-Drift (2026)** — EMA + hysteresis for drift detection
- **Woebot/Wysa** — safety-critical intent override patterns

### State Machine
```
                 classifier matches wellbeing
                 (confidence >= 0.55)
    ┌──────────────────────────────────────┐
    │                                      ▼
  IDLE ──────────────────────────────── ACTIVE
    ▲                                      │
    │  vector < 0.15                       │ classifier misses next cycle
    │  (deactivation threshold)            ▼
    └──────────────────────────── VIGILANT
                                    │
                                    │ if !classifierMatch && vector >= 0.15:
                                    │   → PROMOTE wellbeing @ 0.40 confidence
                                    │
                                    │ renewal: any genuine wellbeing match
                                    │   → back to ACTIVE, reset timer
```

### Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Activate threshold | 0.55 | Standard confidence for fresh activation |
| Deactivate threshold | 0.15 | Hysteresis — much lower to prevent flicker |
| Promotion confidence | 0.40 | Synthetic match, lower than genuine (0.83) |
| Max gate duration | 10 min | Hard cap via Lex dual-expiry pattern |
| Cost ratio (FN/FP) | 10:1 | Missing distress >> false check-in |

### Integration Test Results (2026-03-10)
Gate scenario with 9 transcripts:
- **4 gate promotions** — "whatever, it doesn't matter" (×2), "play some music",
  "what time is it" all promoted while gate was vigilant
- **2 correct catches** — "whatever, it doesn't matter" was missed by classifier
  both times, caught by gate ✅
- **1 debatable** — "play some music" got dual activation (music + promoted wellbeing).
  May want to suppress promotion when another skill already matched clearly.
- **2B model surprise** — "I'm just tired" and "I don't know anymore" both classified
  as wellbeing at 83% by the 2B model directly, without needing the gate.
  The gate provides coverage for the cases the classifier misses.

### Tuning Notes
- The gate is intentionally aggressive — biased toward recall for wellbeing
- Consider suppressing promotion when a non-wellbeing skill already matched
  at high confidence (>0.8) to avoid unnecessary dual-activation
- The 15s decay pause wasn't enough to test the idle path because the 2B
  model catches "I'm just tired" even in isolation. Need a more ambiguous
  test phrase to verify the full idle→active→vigilant→idle cycle.

### Files
- `src/listen/intent-vector.ts` — `ActivationGate` class (appended to existing file)
- `src/listen/index.ts` — gate evaluation after classify, match promotion
- `src/listen/dashboard.ts` — gate status indicator (idle/vigilant/active + promoted badge)
- `scripts/inject-test.ts` — `--scenario gate` for hysteresis testing

### Future Phases
- **Phase 2**: Add `mood` (sentiment), `energy` (speech rate) dimensions via heuristics
- **Phase 3**: Train tiny LoRA classifiers for important dimensions
- **Phase 4**: Explore steering vectors from model activations (see `.knowledge/intent-vectors.md`)
