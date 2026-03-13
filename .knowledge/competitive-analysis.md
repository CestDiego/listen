# Competitive Analysis Notes

Updated: 2026-03-11

## Parakeet (NVIDIA) vs Moonshine (current listen default)

### Findings
- Parakeet has strong WER on LibriSpeech at larger model sizes (notably 0.6B and 1.1B variants).
- Parakeet works on Apple Silicon through `parakeet-mlx` (Python path is viable today).
- Swift-native Parakeet via MLX exists historically but was archived by its maintainers; CoreML path is still evolving.
- Moonshine remains much smaller and more suitable for always-on menubar usage (memory and battery profile).

### Recommendation for listen
- Keep Moonshine as default for real-time always-on transcription in the menubar app.
- Revisit Parakeet only for optional high-accuracy batch/offline file transcription mode.
- Watch for a stable CoreML/ANE-friendly Parakeet path before considering default replacement.

## RCLI (RunanywhereAI) fit analysis

Repo analyzed: `https://github.com/RunanywhereAI/RCLI` (cloned locally to `.tmp/RCLI`).

### What is likely useful for listen
1. **Two-phase execution tracing**
   - Pattern: emit "detected" and then "executed" events for tool/action lifecycle.
   - Value: better debugging of router intent vs actual handler behavior.

2. **Runtime action/skill toggles with persistence**
   - Pattern: users can enable/disable capabilities without code changes.
   - Value: safer rollout for new skills and faster experimentation.

3. **Centralized subprocess execution wrapper**
   - Pattern: normalized timeout/error handling for shell/API bridges.
   - Value: fewer hung processes and better observability in `say`/`afplay`/script calls.

### What is less relevant
- RCLI's terminal-first UX and broad OS-action catalog (listen is focused on mood/audio + wellbeing).
- Heavy C++/MetalRT backend architecture (listen is Bun/TS + Swift + Python experts).

### Recommendation for listen
- Borrow patterns, not architecture.
- Prioritize low-risk improvements:
  - traced decision lifecycle events,
  - skill toggles,
  - shared subprocess runner.
