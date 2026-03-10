# Listen — Architecture

## Skill Expert Models (MLX)

### Overview
Each skill has its own fine-tuned binary classifier running on Apple Silicon via MLX.
Replaces the single 9B LM Studio router with tiny per-skill experts.

### Base Model
**Qwen3.5-0.8B** (Apache 2.0)
- Hybrid architecture: Gated DeltaNet + Gated Attention
- 0.9B params, ~1GB 8-bit on disk
- Native vision-language model, used text-only for classification
- Chat template pre-trained for LoRA PEFT (no embedding fine-tuning needed)

### Training Pipeline (autoresearch-inspired)
```
experts/
├── experts/
│   ├── config.py      # Shared config (base model, skills, defaults)
│   ├── generate.py    # READ-ONLY: data generation + augmentation
│   ├── evaluate.py    # READ-ONLY: eval against test set, compute F1
│   ├── train.py       # AGENT-EDITABLE: LoRA fine-tuning
│   └── serve.py       # HTTP server for Bun integration
├── data/
│   ├── music/         # train.jsonl, valid.jsonl, test.jsonl
│   └── wellbeing/
├── models/
│   ├── music/         # adapters.safetensors/, eval_test.json
│   └── wellbeing/
└── program.md         # Human-editable methodology spec
```

### Results (2026-03-10)

| Skill | Baseline F1 | Fine-tuned F1 | Action Acc | Latency |
|-------|-------------|---------------|------------|---------|
| Music | 94.7% | 100.0% | 77.8% | 278ms |
| Wellbeing | 60.0% | 100.0% | 100.0% | 258ms |

Previous 9B router: 86% accuracy, 1.4s latency.

### Key Learnings

1. **Qwen3.5-0.8B is already strong for classification** — baseline F1 of 94.7% for music
   without any fine-tuning. The model's chat template + system prompt is enough for basic routing.

2. **Class imbalance kills recall** — First wellbeing attempt (37% positive) gave 0% recall.
   Expanding to 42% positive + lower LR (5e-5 vs 1e-4) fixed it completely.

3. **LoRA is incredibly efficient** — 200 iters, batch size 2, 8 layers, rank 8.
   Music trained in 125s, wellbeing in 255s. Peak memory ~5.9GB.

4. **mlx-lm API changes** — `temp` parameter removed from `generate()`.
   Use `sampler=` with `mx.argmax` for greedy decoding instead.

5. **Adapter path is a directory** — `--adapter-path` in mlx-lm creates a directory
   (not a file) containing `adapters.safetensors` + checkpoints.

6. **`--mask-prompt` is essential** — Only trains on the response JSON, not the
   system prompt. Dramatically improves training efficiency for classification.

### Expert vs Router Architecture
```
OLD (single router):
  Transcript → [9B LM Studio] → {skills: [{skill: "music", action: "play"}]}
  ~1.4s per call, 86% accuracy

NEW (parallel experts):
  Transcript → [0.8B music LoRA]    → {match: true, action: "play"}     ~278ms
            → [0.8B wellbeing LoRA] → {match: false}                    ~258ms
  Run in parallel → merge results
  ~300ms total, 100% accuracy
```
