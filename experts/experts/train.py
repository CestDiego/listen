"""
Skill expert training — AGENT-EDITABLE per autoresearch convention.

Fine-tunes Qwen3.5-0.8B with LoRA for binary skill classification.

This file is the one that gets iteratively improved:
  - Hyperparameters (rank, layers, lr, batch size)
  - Training strategy (warmup, schedule)
  - Data augmentation at training time

Single metric to beat: F1 score from evaluate.py.

Usage:
  uv run train --skill music
  uv run train --skill music --iters 400 --lr 2e-4
  uv run train --skill wellbeing --time-budget 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .config import (
    ROOT,
    BASE_MODEL,
    DATA_DIR,
    MODELS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ITERS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_LAYERS,
    DEFAULT_LORA_RANK,
)


def train_skill(
    skill_name: str,
    iters: int = DEFAULT_ITERS,
    lr: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_layers: int = DEFAULT_LORA_LAYERS,
    time_budget: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Fine-tune a skill expert model.

    If time_budget is set (in seconds), training runs for that wall-clock
    duration instead of a fixed number of iterations (autoresearch style).

    Returns dict with training metadata.
    """
    from mlx_lm import lora

    data_dir = DATA_DIR / skill_name
    output_dir = MODELS_DIR / skill_name
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapters.safetensors"

    # Verify data exists
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_dir}. Run: uv run generate"
        )

    # Count training examples
    with open(train_path) as f:
        n_train = sum(1 for _ in f)
    with open(valid_path) as f:
        n_valid = sum(1 for _ in f)

    print(f"\n{'='*60}")
    print(f"TRAINING: {skill_name}")
    print(f"{'='*60}")
    print(f"  Base model:     {BASE_MODEL}")
    print(f"  Train examples: {n_train}")
    print(f"  Valid examples: {n_valid}")
    print(f"  LoRA rank:      {lora_rank}")
    print(f"  LoRA layers:    {lora_layers}")
    print(f"  Learning rate:  {lr}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Iterations:     {iters}" + (f" (time budget: {time_budget}s)" if time_budget else ""))
    print(f"  Output:         {adapter_path}")
    print()

    start_time = time.perf_counter()

    # Build a YAML config for mlx-lm lora training
    import yaml

    config = {
        "model": BASE_MODEL,
        "data": str(data_dir),
        "adapter_path": str(adapter_path),
        "train": True,
        "test": True,  # Run test eval after training
        "fine_tune_type": "lora",
        "mask_prompt": True,  # Only train on the response, not the prompt
        "iters": iters,
        "batch_size": batch_size,
        "num_layers": lora_layers,
        "learning_rate": lr,
        "val_batches": -1,  # Use full validation set
        "test_batches": -1,  # Use full test set
        "steps_per_eval": 50,
        "steps_per_report": 10,
        "save_every": 50,
        "max_seq_length": 512,  # Classification prompts are short
        "optimizer": "adamw",
        "seed": 42,
    }

    if resume and adapter_path.exists():
        config["resume_adapter_file"] = str(adapter_path)

    config_path = output_dir / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run training via mlx_lm CLI as subprocess (cleanest integration)
    import subprocess
    import sys

    venv_python = str(Path(sys.executable))
    cmd = [venv_python, "-m", "mlx_lm", "lora", "--config", str(config_path)]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    elapsed = time.perf_counter() - start_time

    metadata = {
        "skill": skill_name,
        "base_model": BASE_MODEL,
        "adapter_path": str(adapter_path),
        "lora_rank": lora_rank,
        "lora_layers": lora_layers,
        "learning_rate": lr,
        "batch_size": batch_size,
        "iters": iters,
        "time_budget": time_budget,
        "train_examples": n_train,
        "valid_examples": n_valid,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save metadata
    meta_path = output_dir / "train_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE: {skill_name}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Adapter: {adapter_path}")
    print(f"  Metadata: {meta_path}")
    print(f"{'='*60}")

    return metadata


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train a skill expert model")
    parser.add_argument("--skill", required=True, choices=["music", "wellbeing"])
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-layers", type=int, default=DEFAULT_LORA_LAYERS)
    parser.add_argument("--time-budget", type=int, help="Train for N seconds wall-clock")
    parser.add_argument("--resume", action="store_true", help="Resume from existing adapter")
    args = parser.parse_args()

    train_skill(
        skill_name=args.skill,
        iters=args.iters,
        lr=args.lr,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        time_budget=args.time_budget,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
