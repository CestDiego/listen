# Listen — Troubleshooting

## MLX / Fine-tuning

### `generate_step() got unexpected keyword argument 'temp'`
mlx-lm v0.31+ removed `temp` from the generate API. Use a sampler instead:
```python
def greedy_sampler(logits):
    return mx.argmax(logits, axis=-1)
generate(model, tokenizer, prompt=prompt, sampler=greedy_sampler)
```

### Wellbeing model has 0% recall after training
Class imbalance issue. If negatives >> positives, the model learns to always say no.
Fix: expand positive templates until ~40-50% of data is positive. Also try lower LR (5e-5).

### `adapter-path` creates a directory, not a file
When passing `--adapter-path models/skill/adapters.safetensors` to mlx-lm lora,
it creates a *directory* at that path. The actual adapter is inside:
`models/skill/adapters.safetensors/adapters.safetensors`

### Qwen3.5 always includes `<think>` tags
The chat template always wraps with `<think>\n\n</think>\n\n` before the response.
For classification, this is fine — the model outputs JSON after the think tags.
The `parse_output()` function strips them automatically.

## Swift / Menu Bar App

### `No such module 'MoonshineVoice'`
Build with `xcodebuild`, not `swift build`. The MoonshineVoice SPM package
requires proper Xcode project resolution.

### MenuBarExtra content `onAppear` only fires when dropdown opens
Use `MenuBarLabel.onAppear` instead for init logic.

### Studio Display mic outputs 48kHz
Needs `AVAudioConverter` to resample to 16kHz Float32 for Moonshine.

## Bun Pipeline

### SSE connection drops after 10s
Set `idleTimeout: 255` in `Bun.serve()`.

### Shell injection from transcript text
NEVER pass transcript as CLI args. Use temp files + `Bun.spawn` with arg arrays.
