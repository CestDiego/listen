# listen

Live audio observability for your conversations. Transcribes speech locally, watches for patterns you care about, and responds with gentle sounds and voice when it notices something.

```
mic → transcribe → watchlist → gate → analyze → respond
                                  ↓
                           dashboard :3838
```

## What it does

- **Transcribes** speech in real time using [Moonshine](https://github.com/moonshine-ai/moonshine-swift) (100% local, no cloud)
- **Watches** for configurable patterns (e.g., negative self-talk, burnout signals)
- **Gates** with a local LLM to avoid false escalations
- **Analyzes** interesting content with a larger model
- **Responds** with sounds, voice (ElevenLabs or macOS TTS), and notifications
- **Streams** everything to a live web dashboard with inline correction
- **Emits** structured JSONL events for downstream agents

## Architecture

Two processes work together:

| Process | Role |
|---------|------|
| **Bun server** (`--moonshine` mode) | Runs the pipeline: watchlist, gate, analysis, dashboard |
| **Swift menu bar app** | Captures audio via Moonshine, transcribes locally, POSTs to Bun |

The menu bar app shows a mic picker, live partial transcripts, and status indicators.

## Prerequisites

- macOS 13+
- [Bun](https://bun.sh) (`curl -fsSL https://bun.sh/install | bash`)
- [Xcode](https://developer.apple.com/xcode/) (for building the Swift menu bar app)
- [LM Studio](https://lmstudio.ai/) running on `localhost:1234` with a model loaded (for the gate)
- Moonshine model: `pip install moonshine-voice && python -m moonshine_voice.download --language en`

## Quick start

```bash
# 1. Install dependencies
bun install

# 2. Build the menu bar app
bun run listen:menubar:build
bun run listen:menubar:copy

# 3. (Optional) Set up ElevenLabs for voice responses
cp .env.example .env
# Edit .env with your API key

# 4. Start the pipeline
bun run listen:moonshine

# 5. In another terminal, launch the menu bar app
./bin/ListenMenuBar
```

Open http://localhost:3838 to see the live dashboard.

## Modes

| Mode | Command | Description |
|------|---------|-------------|
| `--moonshine` | `bun run listen:moonshine` | Receives transcripts from the menu bar app (recommended) |
| `--pipe` | `bun run listen:pipe` | Paste/pipe transcripts via stdin |
| (default) | `bun run listen` | Legacy: records audio with ffmpeg + transcribes with mlx_whisper |

## Watchlist

The `watchlist.default.json` file contains example patterns. Copy it to customize:

```bash
cp watchlist.default.json watchlist.json
bun run listen:moonshine --watchlist watchlist.json
```

Patterns support string matching and regex, with configurable severity, cooldowns, and response actions (sound + voice message + notification).

## Events

All events are logged to `/tmp/listen-events.jsonl`:

```bash
tail -f /tmp/listen-events.jsonl | jq
```

Event types: `gate.check`, `gate.escalation`, `watchlist.match`, `analysis.complete`

## Options

```
bun run listen:help
```

Key flags:
- `--threshold <0-10>` — gate score to trigger analysis (default: 6)
- `--gate-model <id>` — local LLM for gating (default: glm-4-9b-0414)
- `--gate-endpoint <url>` — LM Studio endpoint (default: http://localhost:1234)
- `--no-watchlist` — disable pattern matching
- `--quiet` — suppress terminal output

## Project structure

```
src/listen/
  index.ts          — CLI entry, 3 modes (live/pipe/moonshine)
  config.ts         — types, CLI parser, defaults
  dashboard.ts      — web UI + SSE + REST API (port 3838)
  session.ts        — timeline store with corrections
  gate.ts           — local LLM gate check
  watchlist.ts      — pattern matcher with cooldowns
  responder.ts      — sound + voice + notification responses
  analyzer.ts       — full analysis via opencode
  buffer.ts         — rolling transcript buffer
  events.ts         — JSONL event emitter
  recorder.ts       — ffmpeg audio capture (legacy mode)
  transcriber.ts    — mlx_whisper transcription (legacy mode)
  notifier.ts       — macOS notifications
  menubar/
    Package.swift   — Swift package (moonshine-swift dependency)
    Sources/ListenMenuBar/
      App.swift     — macOS menu bar app with mic picker
```
