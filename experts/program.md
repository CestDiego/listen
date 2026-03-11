# Autoresearch Program: Multi-Tool Classifier

You are an autonomous ML researcher optimizing a LoRA fine-tune of Qwen3.5-2B
for multi-tool classification. Your goal: maximize **exact-match accuracy** on
the curated eval set (eval-cases.json, ~70 cases).

## Setup (run once at start)

1. Read every file in `experts/experts/` to understand the codebase.
2. Run `uv run python -m experts.experiment --baseline` to establish the baseline.
3. Run `uv run python -m experts.experiment --eval-only` to see current accuracy.
4. Run `uv run python -m experts.experiment --history` to see past experiments.
5. Read `models/multitool/results.tsv` to understand what has been tried.

## File permissions

| File | Permission | Role |
|------|-----------|------|
| `config.py` | READ-ONLY | Model paths, tool definitions |
| `generate_multitool.py` | READ-ONLY | Data templates, system prompt |
| `evaluate_multitool.py` | READ-ONLY | Immutable evaluation function |
| `train_multitool.py` | **AGENT-EDITABLE** | Hyperparams, LoRA config, training loop |
| `experiment.py` | **AGENT-EDITABLE** | Experiment loop, analysis |
| `program.md` | READ-ONLY | These instructions |

## What you CAN do

- Change hyperparameters in `train_multitool.py`:
  - `MULTITOOL_ITERS`, `MULTITOOL_LR`, `MULTITOOL_LORA_RANK`, `MULTITOOL_LORA_LAYERS`
  - `MULTITOOL_BATCH_SIZE`, optimizer settings, sequence length
  - LoRA parameters: rank, alpha, dropout, scale, target layers
- Improve analysis logic in `experiment.py`
- Add new analysis or visualization to `experiment.py`

## What you CANNOT do

- Modify `evaluate_multitool.py` — the eval function is the fixed metric
- Modify `generate_multitool.py` — the training data is fixed infrastructure
- Modify `config.py` — model paths and tool definitions are fixed
- Modify `eval-cases.json` — the ground truth cases are immutable
- Install new packages or add dependencies
- Change the evaluation metric (exact-match accuracy)
- Train for more than 10 minutes wall clock (kill and log as crash)

## The experiment loop

```
LOOP:

1. Read the current state:
   - `uv run python -m experts.experiment --history`
   - Current best accuracy from results.tsv
   - What has been tried before

2. Form a hypothesis:
   - "Higher LoRA rank might capture tool-calling patterns better"
   - "Lower LR might prevent overfitting on the small dataset"
   - "More layers might help with multi-tool decisions"

3. Edit `train_multitool.py` with the change.

4. Run the experiment:
   uv run python -m experts.experiment --iters 50 --note "try rank 32"

5. Read the results:
   - If accuracy improved → status will say "keep"
   - If accuracy same or worse → status will say "discard"

6. Analyze errors:
   - SYSTEMATIC pattern (same error 3+ times) → data problem, not hyperparams
   - SCATTERED errors → need more training or different hyperparams
   - One tool has 0% recall → missing training examples for that tool

7. Log your reasoning, then go to step 1.
```

## Decision framework

- **accuracy improved** → keep. The experiment loop snapshots the best adapter.
- **accuracy same or worse** → discard. Revert your change to `train_multitool.py`.
- **crash** → if easy fix (typo, OOM), fix and retry. If fundamental, log and move on.
- **systematic errors** → this is a DATA problem, not a hyperparameter problem.
  Flag it for the human. Do not try to fix data problems with hyperparameter tuning.

## Simplicity criterion

All else being equal, simpler is better:
- 0.5% improvement with 20 lines of new code? Probably not worth it.
- 0.5% improvement from removing code? Definitely keep.
- Equal accuracy with simpler config? Keep.

## Current search space

These are the knobs in `train_multitool.py`:

```
MULTITOOL_ITERS = 300        # try: 50, 100, 200, 500
MULTITOOL_LR = 5e-5          # try: 1e-5, 2e-5, 5e-5, 1e-4, 2e-4
MULTITOOL_LORA_RANK = 16     # try: 4, 8, 16, 32, 64
MULTITOOL_LORA_LAYERS = 12   # try: 4, 8, 12, 16, 24
```

Plus LoRA alpha, dropout, and optimizer settings in the config dict.

## Notes

- Training data: ~3,800 examples (40% positive, 60% negative)
- 5 tools: accommodator.{activate, skip, deactivate, set_target}, wellbeing.check_in
- Eval set: ~70 curated cases covering all tools + negatives + dual-activation
- Each 50-iter experiment takes ~3 minutes total (train + eval)
- The model is Qwen3.5-2B (chat format with native `<tool_call>` tokens)
- `--mask-prompt` is essential: only train on the assistant response
