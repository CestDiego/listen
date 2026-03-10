# listen

Live audio observability for your conversations. Transcribes speech locally, watches for patterns you care about, and responds with gentle sounds and voice when it notices something.

Everything runs on your Mac — no audio leaves your machine.

```
mic → transcribe → watchlist → gate → analyze → respond
                                  ↓
                           dashboard :3838
```

## Setup (macOS only)

### 1. Install prerequisites

```bash
# Bun (JS/TS runtime)
curl -fsSL https://bun.sh/install | bash

# Xcode (needed to build the menu bar app)
# Install from the App Store, then accept the license:
sudo xcodebuild -license accept

# Python + Moonshine model (local speech-to-text)
pip install moonshine-voice
python -m moonshine_voice.download --language en

# LM Studio (local LLM for the gate check)
# Download from https://lmstudio.ai
# Open it, search for "glm-4-9b-0414", download, and start the server
```

### 2. Clone and build

```bash
git clone https://github.com/CestDiego/listen.git
cd listen

# Install JS dependencies
bun install

# Build the Swift menu bar app (~60s first time, downloads Moonshine framework)
./start.sh --build
```

### 3. (Optional) Set up ElevenLabs for voice responses

Without this, voice responses use macOS `say` (which still works fine).

```bash
cp .env.example .env
# Edit .env with your ElevenLabs API key
```

### 4. Run

```bash
./start.sh
```

This launches both processes:
- **Bun pipeline** — watchlist, gate, analysis, web dashboard
- **Menu bar app** — captures audio, transcribes with Moonshine, posts to Bun

You'll see a `listen` icon appear in your menu bar. Click it to see the mic picker, live transcripts, and status.

Open **http://localhost:3838** for the web dashboard.

### Stopping

```bash
./start.sh --stop
```

Or just `Ctrl+C` in the terminal where `start.sh` is running.

## What it does

- **Transcribes** speech in real time using [Moonshine](https://github.com/moonshine-ai/moonshine-swift) (100% local, no cloud)
- **Watches** for configurable patterns (e.g., negative self-talk, burnout signals)
- **Gates** with a local LLM to avoid false escalations
- **Analyzes** interesting content with a larger model when the gate score is high enough
- **Responds** with sounds, voice (ElevenLabs or macOS TTS), and macOS notifications
- **Streams** everything to a live web dashboard with inline transcript correction
- **Emits** structured JSONL events for downstream agents

## Architecture

```
┌─────────────────────────┐     POST /api/transcript     ┌──────────────────────────┐
│  Swift Menu Bar App     │ ──────────────────────────▶  │  Bun listen process       │
│  (Moonshine transcriber)│                               │  (gate/watchlist/analyze) │
│  - Mic picker           │  ◀────────────────────────── │  - Dashboard :3838        │
│  - Live partial text    │     SSE /events               │  - Session store          │
│  - Status indicator     │                               │  - Event log (JSONL)      │
└─────────────────────────┘                               └──────────────────────────┘
```

## Watchlist

The `watchlist.default.json` file contains example patterns. Copy it to customize:

```bash
cp watchlist.default.json watchlist.json
# Edit watchlist.json — add your own patterns
./start.sh  # will use watchlist.default.json by default
```

Patterns support string matching and regex, with configurable severity, cooldowns, and response actions (sound + voice message + notification).

## Events

All events are logged to `/tmp/listen-events.jsonl`:

```bash
tail -f /tmp/listen-events.jsonl | jq
```

Event types: `gate.check`, `gate.escalation`, `watchlist.match`, `analysis.complete`

## CLI options

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
start.sh              — one-command launcher (builds, starts, stops)
watchlist.default.json — example watchlist patterns
src/listen/
  index.ts            — CLI entry, 3 modes (live/pipe/moonshine)
  config.ts           — types, CLI parser, defaults
  dashboard.ts        — web UI + SSE + REST API (port 3838)
  session.ts          — timeline store with corrections
  gate.ts             — local LLM gate check
  watchlist.ts        — pattern matcher with cooldowns
  responder.ts        — sound + voice + notification responses
  analyzer.ts         — full analysis via opencode
  buffer.ts           — rolling transcript buffer
  events.ts           — JSONL event emitter
  recorder.ts         — ffmpeg audio capture (legacy mode)
  transcriber.ts      — mlx_whisper transcription (legacy mode)
  notifier.ts         — macOS notifications
  menubar/
    Package.swift     — Swift package (moonshine-swift dependency)
    Sources/ListenMenuBar/
      App.swift       — macOS menu bar app with mic picker
```
