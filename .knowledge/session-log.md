# Listen — Session Log

## Current State (updated 2026-03-10)
- **Branch**: `main`
- **Commits ahead of origin**: 12
- **Last commit**: `7db86ff` feat: NRC VAD Lexicon for mood (valence) and energy (arousal) — Phase 3a
- **Typecheck**: clean (working tree clean, no uncommitted changes)
- **Blockers**: none — ready to push or continue building

## What We Just Did
- Implemented **NRC VAD Lexicon** integration (44,728 unigrams) for mood (valence) and energy (arousal) — Phase 3a
- Added Phase 3 dataset research: 70+ public datasets across all 6 intent vector dimensions (`.knowledge/datasets.md`)
- Built **extensible dimension system** (Phase 2): mood + energy as computed dimensions with config-driven registry (`DIMENSION_DEFS`)
- Implemented **activation gate** with Schmitt-trigger hysteresis for post-classification wellbeing bias
- Added **intent vector visualization**: live radar chart, sparklines, dimension readouts in SSE-driven dashboard
- Upgraded classifier to **Qwen3.5-2B** multi-tool model — 98.4% accuracy (63/64), up from 95.5% on 22-case set
- Evolved from N binary experts → single unified multi-tool classifier with `<tool_call>` token support
- Added comprehensive `.knowledge/` documentation (architecture, datasets, intent-vectors, troubleshooting)

## Immediate Next Steps
1. **Push to origin** — 12 commits ahead, all clean. `git push` when ready.
2. **Phase 3 dataset training** — download top-priority datasets (CLINC150, MASSIVE, Snips, GoEmotions) and augment the 291-example training set
3. **Fix remaining eval failure** — "not good enough" triggers spurious `music.skip` alongside correct `wellbeing.check_in`. Add negative reinforcement examples.
4. **Gate suppression logic** — suppress wellbeing promotion when another skill already matched at high confidence (>0.8) to avoid unnecessary dual-activation
5. **Phase 3 LoRA classifiers** — train tiny LoRA classifiers for important computed dimensions (mood, energy) to replace/augment heuristic computation
6. **Phase 4** — explore steering vectors from model activations (research in `.knowledge/intent-vectors.md`)
7. **Phase 5** — multiple activation gates for different skills (gate is already parameterized via `GateConfig`)

## Architecture Quick Reference

### Pipeline Flow
```
Swift MenuBar (Moonshine transcription, mic picker)
  → Bun Pipeline (src/listen/index.ts)
    → classifyTranscript() → serve_multitool.py → Qwen3.5-2B + LoRA
      → parse <tool_call> blocks → VALID_TOOLS filter
      → {skills: [{skill, action, confidence}]}
    → ActivationGate evaluation (hysteresis: idle→active→vigilant→idle)
    → IntentVectorStore.update() (decay + activate + compute hooks)
    → Skill execution (music.ts, wellbeing via ElevenLabs)
    → SSE events → Dashboard (radar chart, sparklines, gate indicator)
```

### Key Files
| File | Purpose |
|------|---------|
| `src/listen/index.ts` | Main pipeline orchestration |
| `src/listen/intent-vector.ts` | `DIMENSION_DEFS`, `IntentVectorStore`, `ActivationGate` |
| `src/listen/dashboard.ts` | SSE-driven web dashboard (radar, sparklines, decision log) |
| `src/listen/nrc-vad.ts` | NRC VAD Lexicon loader — mood (valence) + energy (arousal) |
| `src/listen/skills/classify.ts` | HTTP client for MLX classifier |
| `src/listen/skills/router.ts` | Skill router (post-classify dispatch) |
| `src/listen/session.ts` | Session state, SSE emission, stats |
| `experts/experts/serve_multitool.py` | MLX model server (single Qwen3.5-2B + LoRA) |
| `experts/experts/train_multitool.py` | Unified LoRA fine-tuning pipeline |
| `experts/experts/generate_multitool.py` | Training data gen with dedup + stratified split |
| `scripts/inject-test.ts` | Test injection (9 transcripts, timed sequence) |
| `start.sh` | Single-command launcher (pipeline + expert server + dashboard) |

### 6-Axis Intent Vector
| Dimension | Range | Source | Half-Life | How |
|-----------|-------|--------|-----------|-----|
| `music` | [0,1] | classifier | 45s | Confidence on match (0.95) |
| `wellbeing` | [0,1] | classifier | 120s | Confidence on match (0.83-0.95) |
| `engagement` | [0,1] | computed | 60s | Chunks in last 60s / 12 |
| `taskFocus` | [0,1] | computed | 30s | Skill-matched chunk ratio |
| `mood` | [-1,1] | computed | 120s | NRC VAD Lexicon avg valence |
| `energy` | [0,1] | computed | 60s | 0.6×NRC arousal + 0.4×speech rate |

### Classifier
- **Model**: Qwen3.5-2B (Apache 2.0), LoRA rank 16, 12 layers
- **Training data**: 291 entries (202 train / 42 valid / 47 test)
- **Accuracy**: 98.4% (63/64), dual-activation 4/4, negatives 34/35
- **Latency**: ~368ms avg on Apple Silicon
- **Tools**: `music.play/pause/resume/skip/previous/volume_up/volume_down`, `wellbeing.check_in`

## Active Decisions
- **[0,1] range for all dimensions** — classifier softmax outputs are natively [0,1]; decay math stays clean; composability (mood × energy stays in range); NRC VAD Lexicon speaks [0,1]. Dashboard formats for display.
- **NRC VAD Lexicon over ML sentiment** — 44,728 unigrams with continuous V/A/D scores. Zero latency, no model loading, good enough for Phase 3a. ML models can replace later.
- **Unified multi-tool classifier over per-skill experts** — single 2B model with tool definitions in system prompt beats N binary experts on accuracy and operational simplicity.
- **Activation gate is aggressive on wellbeing** — FN/FP cost ratio 10:1. Missing distress >> false check-in. Gate promotes at 0.40 confidence when vigilant.
- **Config-driven dimensions** — `DIMENSION_DEFS` is the single source of truth. Engine, gate, and dashboard all derive from it. Adding a dimension = one config entry.

## Known Issues / Tech Debt
- **1 eval failure**: "not good enough" triggers spurious `music.skip` alongside correct `wellbeing.check_in` (over-activation)
- **Gate dual-activation ambiguity**: gate promotes wellbeing even when another skill already matched clearly — consider suppressing when non-target skill > 0.8 confidence
- **15s decay pause insufficient for gate test**: 2B model catches "I'm just tired" in isolation, so idle→active→vigilant→idle cycle can't be fully verified with current test phrases
- **Legacy per-skill pipeline still present**: `generate.py`, `train.py`, `evaluate.py`, `serve.py` + per-skill adapters in `models/music/`, `models/wellbeing/` — works as fallback but adds maintenance surface
- **Qwen3.5 `<think>` tags**: model always emits `<think>\n\n</think>` before response — harmless but parser must strip them
- **SSE idle timeout**: must set `idleTimeout: 255` in `Bun.serve()` or connections drop after 10s
- **Shell injection risk**: transcript text must never be passed as CLI args — use temp files + `Bun.spawn` with arg arrays

## Dataset Status
**Downloaded / integrated:**
- NRC VAD Lexicon v2.1 — 44,728 unigrams, used for mood (valence) + energy (arousal)

**Researched, not yet downloaded:**
| Dataset | Size | Primary Use | Priority |
|---------|------|-------------|----------|
| CLINC150-OOS | 23.9k | Intent classification + negatives | High |
| MASSIVE (en-US) | 11.5k | Music/audio/weather intents + slots | High |
| Snips NLU | ~14k | PlayMusic + AddToPlaylist with slots | High |
| GoEmotions | 58k | Mood/wellbeing + negatives (27 emotions) | High |
| EmpatheticDialogues | 107k | Wellbeing/distress detection | Medium |
| DailyDialog | 102k | Mood + engagement + negatives | Medium |
| MELD | 13.7k | Mood/arousal in dialogue | Medium |
| EmoBank | 10k | Continuous VAD (valence-arousal-dominance) | Medium |
| BANKING77 | 13k | Pure negatives (banking queries) | Medium |
| ESConv | 31k | Emotional support / wellbeing | Medium |
| SLURP | 72k | Spoken intents (music/audio hierarchy) | Low |
| HWU64 | 25.7k | 64 intents including music/audio | Low |

## Commit History (Unpushed — 12 commits)
```
7db86ff feat: NRC VAD Lexicon for mood (valence) and energy (arousal) — Phase 3a
ce15df9 docs: add Phase 3 dataset research (70+ public datasets across all dimensions)
eb4d842 feat: extensible dimension system with mood + energy (Phase 2)
204e3ff docs: add activation gate architecture and integration test results
22221db feat: activation gate with hysteresis for post-classification wellbeing bias
2e71c1e feat: intent vector visualization with live radar chart in dashboard
17550c3 feat: upgrade to Qwen3.5-2B multi-tool model (98.4% accuracy)
c38c115 feat: unified multi-tool classifier replaces N binary experts
66e5307 docs: update architecture with parallel inference findings
73a10dd feat: true parallel inference via worker processes (~2× speedup)
d66495f feat: replace router with parallel expert classifier + per-expert observability
573b815 feat: replace LM Studio with self-contained MLX expert models
```

**Diff stat**: 61 files changed, 8,479 insertions(+), 623 deletions(-)
