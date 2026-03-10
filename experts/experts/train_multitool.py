"""
Multi-tool model training — AGENT-EDITABLE per autoresearch convention.

Fine-tunes a single Qwen3.5-0.8B with LoRA for multi-tool calling.
Unlike the per-skill binary classifiers, this trains ONE model that
outputs 0..N <tool_call> blocks per transcript.

Single metric to beat: exact-match accuracy from evaluate_multitool.py.

Usage:
  uv run train-multitool
  uv run train-multitool --iters 400 --lr 2e-4
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
    DEFAULT_LORA_LAYERS,
    DEFAULT_LORA_RANK,
)

# ── Tuned defaults for multi-tool ─────────────────────────────────
# Multi-tool output is longer than binary JSON, so we need more
# sequence length and potentially more iterations.
MULTITOOL_ITERS = 300
MULTITOOL_LR = 5e-5       # Lower LR — multi-tool output is more complex
MULTITOOL_LORA_RANK = 16   # Higher rank for richer tool-calling patterns
MULTITOOL_LORA_LAYERS = 12 # More layers — tool calling uses deeper representations


def train_multitool(
    iters: int = MULTITOOL_ITERS,
    lr: float = MULTITOOL_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lora_rank: int = MULTITOOL_LORA_RANK,
    lora_layers: int = MULTITOOL_LORA_LAYERS,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Fine-tune the unified multi-tool model.

    Returns dict with training metadata.
    """
    data_dir = DATA_DIR / "multitool"
    output_dir = MODELS_DIR / "multitool"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapters.safetensors"

    # Verify data exists
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_dir}. Run: uv run generate-multitool"
        )

    # Count training examples
    with open(train_path) as f:
        n_train = sum(1 for _ in f)
    with open(valid_path) as f:
        n_valid = sum(1 for _ in f)

    print(f"\n{'='*60}")
    print(f"TRAINING: multi-tool unified model")
    print(f"{'='*60}")
    print(f"  Base model:     {BASE_MODEL}")
    print(f"  Train examples: {n_train}")
    print(f"  Valid examples: {n_valid}")
    print(f"  LoRA rank:      {lora_rank}")
    print(f"  LoRA layers:    {lora_layers}")
    print(f"  Learning rate:  {lr}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Iterations:     {iters}")
    print(f"  Output:         {adapter_path}")
    print()

    start_time = time.perf_counter()

    # Build YAML config for mlx-lm lora training
    import yaml

    config = {
        "model": BASE_MODEL,
        "data": str(data_dir),
        "adapter_path": str(adapter_path),
        "train": True,
        "test": True,
        "fine_tune_type": "lora",
        "mask_prompt": True,  # Only train on the assistant response
        "iters": iters,
        "batch_size": batch_size,
        "num_layers": lora_layers,
        "learning_rate": lr,
        "val_batches": -1,
        "test_batches": -1,
        "steps_per_eval": 50,
        "steps_per_report": 10,
        "save_every": 50,
        "max_seq_length": 1024,  # Longer — multi-tool output can be verbose
        "optimizer": "adamw",
        "seed": 42,
        "lora_parameters": {
            "rank": lora_rank,
            "alpha": lora_rank * 2,  # Standard alpha = 2 * rank
            "dropout": 0.05,
            "scale": 1.0,
        },
    }

    if resume and adapter_path.exists():
        config["resume_adapter_file"] = str(adapter_path)

    config_path = output_dir / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run training via mlx_lm CLI
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
        "model_type": "multitool",
        "base_model": BASE_MODEL,
        "adapter_path": str(adapter_path),
        "lora_rank": lora_rank,
        "lora_layers": lora_layers,
        "learning_rate": lr,
        "batch_size": batch_size,
        "iters": iters,
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
    print(f"TRAINING COMPLETE: multi-tool")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Adapter: {adapter_path}")
    print(f"  Metadata: {meta_path}")
    print(f"{'='*60}")

    return metadata


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train multi-tool model")
    parser.add_argument("--iters", type=int, default=MULTITOOL_ITERS)
    parser.add_argument("--lr", type=float, default=MULTITOOL_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lora-rank", type=int, default=MULTITOOL_LORA_RANK)
    parser.add_argument("--lora-layers", type=int, default=MULTITOOL_LORA_LAYERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_multitool(
        iters=args.iters,
        lr=args.lr,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
