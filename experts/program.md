# Skill Expert Training — Program

Following [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) methodology.

## Architecture

Each skill gets its own **binary expert model**: a LoRA-fine-tuned Qwen3.5-0.8B that answers one question: "Should this skill fire for this transcript?"

```
Transcript → [music expert] → {"match": true, "action": "play"}
          → [wellbeing expert] → {"match": false}
```

Experts run in parallel. Results merge. This replaces the single 9B LM Studio router.

## Three-File Pattern

| File | Role | Editable? |
|------|------|-----------|
| `experts/generate.py` | Data generation + augmentation | Read-only (infrastructure) |
| `experts/evaluate.py` | Eval against test set, compute F1 | Read-only (tamper-proof) |
| `experts/train.py` | LoRA fine-tuning | Agent-editable (iterate here) |

## Single Metric

**F1 score** per skill expert. Balances precision (don't fire when you shouldn't) and recall (don't miss when you should).

Baseline (Qwen3.5-0.8B without fine-tuning) establishes the floor.
Target: F1 > 0.90 for both skills.

## Keep/Discard Loop

1. Run baseline eval → record F1
2. Train with current hyperparams → record F1
3. If F1 improved → keep (git commit)
4. If F1 same or worse → revert, try different hyperparams
5. Repeat until time budget exhausted or target reached

## Workflow

```bash
# 1. Generate training data
uv run generate

# 2. Run baseline eval (no fine-tuning)
uv run evaluate --skill music
uv run evaluate --skill wellbeing

# 3. Train
uv run train --skill music
uv run train --skill wellbeing

# 4. Eval with adapter
uv run evaluate --skill music --adapter models/music/adapters.safetensors
uv run evaluate --skill wellbeing --adapter models/wellbeing/adapters.safetensors

# 5. Serve (integration with Bun pipeline)
uv run serve --port 8234
```

## Base Model

**Qwen3.5-0.8B** (Apache 2.0)
- 0.9B params, ~1GB on disk (8-bit)
- Gated DeltaNet + Gated Attention hybrid architecture
- Chat template pre-trained for LoRA PEFT
- Runs fast on Apple Silicon via MLX

## Update Log

- 2026-03-10: Initial program document created
