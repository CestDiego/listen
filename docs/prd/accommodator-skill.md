# Mood Accommodator Skill — PRD

## Update Log

- 2026-03-10: Initial draft
- 2026-03-10: **Implementation complete** — all 5 phases implemented across parallel worktrees:
  - Phase 1: Playlist infrastructure (config.yaml, download script, directory scaffolding)
  - Phase 2: Audio engine (afplay/ffplay backend with crossfade, queue management, 19 tests)
  - Phase 3: Accommodator skill (circumplex mapping, ISO state machine, intent vector integration, 16 tests)
  - Phase 4: Classifier retrain (eval cases remapped, classify/router/types updated from music.* to accommodator.*)
  - Phase 5: Dashboard (Accommodator tab with circumplex quadrant display, ISO progress bar, SSE emission)
  - Post-review fixes: crossfade race condition, subprocess leak on shutdown, getState() numeric field passthrough, playlist drift reload

## Overview

The Mood Accommodator is a new skill that watches the intent vector's `mood`, `energy`, and `wellbeing` dimensions and proactively adjusts the user's audio environment to steer mood. It replaces direct music commands as the primary music interaction model — music becomes a **downstream output of user state**, not a first-class intent.

Audio is sourced from **pre-downloaded mood playlists** stored on the local filesystem (privacy-first, no streaming dependency). Playlists are curated from YouTube, downloaded via `yt-dlp`, and organized by mood quadrant using Russell's Circumplex Model of Affect (valence x arousal).

The skill implements the **ISO principle** from music therapy: instead of jumping directly to a target mood's playlist, it first matches the user's current emotional state, then gradually transitions toward the desired state over time.

## Problem Statement

The current system treats music as a direct command target (`music.play`, `music.pause`, `music.skip`). This creates two problems:

1. **Friction**: The user must explicitly request music changes. There's no proactive audio adjustment based on detected state.
2. **Missed therapeutic opportunity**: Music therapy research shows that matching-then-steering (ISO principle) is more effective at mood regulation than playing "happy music" for a sad person.
3. **Classifier noise**: `music.*` tool definitions compete with `wellbeing.check_in` in ambiguous cases (e.g., "not good enough" triggers spurious `music.skip` — the current remaining eval failure).

The Accommodator unifies these concerns: the user's state drives audio, the classifier is simplified, and the system can proactively support the user's emotional wellbeing through sound.

## Goals & Non-Goals

### Goals

- **G1**: Implement a state-driven audio system that responds to intent vector changes
- **G2**: Support opt-in/opt-out activation (user must consent to mood steering)
- **G3**: Use pre-downloaded playlists organized by Russell's Circumplex quadrants
- **G4**: Implement ISO-principle transitions (match current state, then steer)
- **G5**: Remove `music.*` from the classifier tool definitions to reduce false positives
- **G6**: Provide a playlist download script for curating local audio content
- **G7**: Integrate with the existing dashboard for visibility into Accommodator state

### Non-Goals

- **NG1**: Streaming music integration (Apple Music, Spotify) — this is intentionally local-only for v1
- **NG2**: AI music generation (noted as future direction, but not v1 scope)
- **NG3**: Replacing the existing `music.ts` skill entirely — it stays as a legacy/override path
- **NG4**: Multi-user support — single-user desktop system
- **NG5**: Sophisticated audio analysis (BPM detection, spectral features) — metadata is best-effort

## Architecture

### System Context

```
                     ┌──────────────────────────────────────────────────┐
                     │              Listen Pipeline                     │
                     │                                                  │
Transcript ─────────►│  Classifier ──► IntentVectorStore.update()       │
                     │                       │                          │
                     │              mood [-1,+1]  energy [0,1]          │
                     │              wellbeing [0,1]  taskFocus [0,1]    │
                     │                       │                          │
                     │              ┌────────▼────────┐                 │
                     │              │  Accommodator    │                 │
                     │              │  (watches SSE)   │                 │
                     │              │                  │                 │
                     │              │  current state ──┼──► quadrant    │
                     │              │  target state  ──┼──► ISO path    │
                     │              │  playlist pick ──┼──► AudioEngine │
                     │              └─────────────────-┘                │
                     │                       │                          │
                     │              ┌────────▼────────┐                 │
                     │              │  AudioEngine     │                 │
                     │              │  (afplay/ffplay) │                 │
                     │              │  crossfade,vol   │                 │
                     │              └──────────────────┘                │
                     │                                                  │
                     │              Dashboard SSE ──► "Accommodator" tab│
                     └──────────────────────────────────────────────────┘
```

### Key Architectural Decision: Music as Downstream Output

**Before (current):**
```
User says "play some music" → Classifier → music.play → Apple Music
User says "I'm stressed"    → Classifier → wellbeing.check_in → voice response
(No connection between mood and music)
```

**After (Accommodator):**
```
User says "help me focus"      → Classifier → accommodator.activate
User's mood/energy detected    → IntentVector → Accommodator watches
Accommodator selects playlist  → "focus" playlist → AudioEngine plays locally
Mood shifts over time          → ISO transition → crossfade to new quadrant
User says "play some jazz"     → Classifier → accommodator.set_target (override)
User says "stop the music"     → Classifier → accommodator.deactivate
```

The `music.*` tool definitions are removed from the classifier. Direct music commands route through the Accommodator instead. The existing `music.ts` (Apple Music via AppleScript) remains as importable code but is no longer a classifier target.

## Detailed Design

### Playlist System

#### Directory Structure

```
data/playlists/
├── config.yaml              # Playlist source URLs + metadata
├── uplift/                  # High energy + positive mood
│   ├── _meta.json           # Playlist-level metadata
│   ├── 001-track-title.opus
│   ├── 002-track-title.opus
│   └── ...
├── release/                 # High energy + negative mood
│   ├── _meta.json
│   └── ...
├── calm/                    # Low energy + positive mood
│   ├── _meta.json
│   └── ...
├── comfort/                 # Low energy + negative mood
│   ├── _meta.json
│   └── ...
├── focus/                   # High taskFocus (any mood)
│   ├── _meta.json
│   └── ...
└── neutral/                 # No strong signal / background
    ├── _meta.json
    └── ...
```

#### Track Metadata (`_meta.json`)

```json
{
  "quadrant": "uplift",
  "description": "Upbeat, energetic, happy — for lifting mood with high energy",
  "tracks": [
    {
      "filename": "001-walking-on-sunshine.opus",
      "title": "Walking on Sunshine",
      "artist": "Katrina and the Waves",
      "duration_seconds": 238,
      "bpm": 110,
      "source_url": "https://youtube.com/watch?v=...",
      "downloaded_at": "2026-03-10T12:00:00Z",
      "mood_tags": ["happy", "energetic", "uplifting"]
    }
  ]
}
```

#### Playlist Downloader Script

**File**: `scripts/download-playlists.ts` (Bun/TypeScript — consistent with project tooling)

```typescript
#!/usr/bin/env bun
/**
 * Download mood playlists from YouTube using yt-dlp.
 *
 * Usage:
 *   bun run scripts/download-playlists.ts                    # download all
 *   bun run scripts/download-playlists.ts --quadrant uplift  # download one quadrant
 *   bun run scripts/download-playlists.ts --dry-run          # show what would be downloaded
 *
 * Prerequisites:
 *   brew install yt-dlp ffmpeg
 */

import { readFile, writeFile, mkdir } from "fs/promises";
import { resolve, join } from "path";
import { parse as parseYaml } from "yaml";  // bun add yaml

interface PlaylistSource {
  quadrant: string;
  description: string;
  urls: string[];      // YouTube playlist/video URLs
  queries: string[];   // YouTube search queries (yt-dlp "ytsearch5:query")
}

interface TrackMeta {
  filename: string;
  title: string;
  artist: string;
  duration_seconds: number;
  bpm: number | null;
  source_url: string;
  downloaded_at: string;
  mood_tags: string[];
}

const DATA_DIR = resolve(import.meta.dir, "../data/playlists");
const CONFIG_PATH = join(DATA_DIR, "config.yaml");

async function downloadQuadrant(source: PlaylistSource): Promise<void> {
  const dir = join(DATA_DIR, source.quadrant);
  await mkdir(dir, { recursive: true });

  // Load existing metadata to skip already-downloaded tracks
  const metaPath = join(dir, "_meta.json");
  let existing: TrackMeta[] = [];
  try {
    const raw = await readFile(metaPath, "utf-8");
    existing = JSON.parse(raw).tracks || [];
  } catch { /* first run */ }

  const existingUrls = new Set(existing.map(t => t.source_url));

  for (const url of source.urls) {
    if (existingUrls.has(url)) {
      console.log(`  skip (already downloaded): ${url}`);
      continue;
    }

    // yt-dlp: download audio only, convert to opus, embed metadata
    const proc = Bun.spawn([
      "yt-dlp",
      "--extract-audio",
      "--audio-format", "opus",
      "--audio-quality", "5",      // good quality, ~96kbps
      "--output", join(dir, "%(autonumber)03d-%(title)s.%(ext)s"),
      "--no-playlist",             // one video at a time (playlists handled by URL list)
      "--write-info-json",         // metadata sidecar for duration/title
      "--no-overwrites",
      url,
    ], { stdout: "pipe", stderr: "pipe" });

    const exitCode = await proc.exited;
    if (exitCode !== 0) {
      const stderr = await new Response(proc.stderr).text();
      console.error(`  FAIL: ${url} — ${stderr.trim()}`);
    }
  }

  // After download, scan directory and rebuild _meta.json
  // (implementation reads .info.json sidecars for metadata)
}
```

**Key behaviors:**
- **Idempotent**: skips already-downloaded tracks (checks `_meta.json`)
- **Audio-only**: uses `yt-dlp --extract-audio --audio-format opus`
- **Metadata**: reads `.info.json` sidecars from `yt-dlp --write-info-json`
- **BPM detection**: optional post-processing with `ffmpeg` + aubio or similar (best-effort, stored as `null` if unavailable)
- **Config-driven**: playlist URLs come from `data/playlists/config.yaml`

### Mood Mapping (Circumplex)

Based on Russell's Circumplex Model of Affect, we map the 2D space of `mood` (valence, x-axis) and `energy` (arousal, y-axis) to six playlist regions:

```
        energy
        1.0 ┌─────────────────────────┐
            │  RELEASE    │  UPLIFT   │
            │  (intense,  │  (upbeat, │
            │  cathartic) │  happy)   │
        0.5 ├─────────────┼───────────┤
            │  COMFORT    │  CALM     │
            │  (gentle,   │  (ambient,│
            │  warm)      │  peaceful)│
        0.0 └─────────────┴───────────┘
           -1.0    mood=0.0     mood=1.0

  Special overlays:
    taskFocus > 0.6  →  "focus" (overrides quadrant)
    |mood| < 0.15 && energy < 0.4  →  "neutral" (no strong signal)
```

#### Quadrant Selection Logic

```typescript
interface QuadrantSelection {
  quadrant: "uplift" | "release" | "calm" | "comfort" | "focus" | "neutral";
  confidence: number;  // 0-1, how strongly we're in this quadrant
}

function selectQuadrant(
  mood: number,      // [-1, +1]
  energy: number,    // [0, 1]
  taskFocus: number, // [0, 1]
): QuadrantSelection {
  // Override: high task focus → focus playlist regardless of mood
  if (taskFocus > 0.6) {
    return { quadrant: "focus", confidence: taskFocus };
  }

  // Neutral zone: no strong signal
  if (Math.abs(mood) < 0.15 && energy < 0.4) {
    return { quadrant: "neutral", confidence: 1 - Math.abs(mood) * 3 };
  }

  // Circumplex quadrants
  const isPositive = mood >= 0;
  const isHighEnergy = energy >= 0.5;

  if (isPositive && isHighEnergy)  return { quadrant: "uplift",  confidence: mood * energy };
  if (!isPositive && isHighEnergy) return { quadrant: "release", confidence: Math.abs(mood) * energy };
  if (isPositive && !isHighEnergy) return { quadrant: "calm",    confidence: mood * (1 - energy) };
  return { quadrant: "comfort", confidence: Math.abs(mood) * (1 - energy) };
}
```

### ISO Principle Algorithm

The ISO (Iso-Moodic) principle from music therapy states: to change someone's mood through music, you must first **match** their current state, then **gradually lead** them toward the target state.

#### State Machine

```
         ┌─────────┐    user activates     ┌──────────┐
         │ INACTIVE ├─────────────────────► │ MATCHING │
         └────▲─────┘                       └────┬─────┘
              │                                  │
              │ user deactivates            mood matches
              │                            current quadrant
              │                            for >= matchDuration
              │                                  │
              │                                  ▼
              │                            ┌───────────┐
              │                            │ STEERING  │
              │                            │           │
              │                            │ gradually │
              │                            │ shift     │
              └────────────────────────────┤ quadrant  │
                                           └───────────┘
```

#### ISO Transition Parameters

```typescript
interface ISOConfig {
  /** How long to stay in "matching" phase before beginning to steer (ms). */
  matchDurationMs: number;         // default: 5 * 60_000 (5 minutes)

  /** How long each steering step takes before selecting next quadrant (ms). */
  steerStepMs: number;             // default: 10 * 60_000 (10 minutes)

  /** Minimum mood change per steering step (prevents oscillation). */
  minMoodDelta: number;            // default: 0.2

  /** If user's mood has already improved this much, skip steering. */
  moodImprovedThreshold: number;   // default: 0.3
}
```

#### Transition Path Example

User is sad + low-energy (mood = -0.6, energy = 0.2):

1. **MATCHING** (0-5 min): Play from "comfort" playlist (gentle, warm, soothing) — matches current state
2. **STEERING step 1** (5-15 min): Crossfade to "calm" playlist (ambient, peaceful) — same energy, higher valence
3. **STEERING step 2** (15-25 min): If mood has improved, crossfade to "uplift" (or stay at "calm" if mood hasn't moved much)
4. **ARRIVED**: If mood > 0.2 and energy > 0.3, hold current playlist

The path always moves through **adjacent quadrants** — never jumps from "comfort" directly to "uplift" (which would skip "calm"). Adjacency:

```
comfort ↔ calm ↔ uplift
comfort ↔ release ↔ uplift
```

#### Target State

By default, the target is "neutral-positive" (mood > 0, moderate energy). But the user can override via `accommodator.set_target`:

```
"I want to feel calm"     → target: calm quadrant
"help me get energized"   → target: uplift quadrant
"I need to focus"         → target: focus (overrides circumplex)
```

### Accommodator Skill Interface

**File**: `src/listen/skills/accommodator.ts`

This skill implements the existing `Skill` interface from `src/listen/skills/types.ts`:

```typescript
import type { Skill, SkillResponse, RouterContext } from "./types";

// ── Existing Skill interface (from types.ts) for reference ────────
//
// interface Skill {
//   name: string;
//   description: string;
//   actions: SkillAction[];
//   hints?: RegExp[];
//   handle: (action: string, params: Record<string, string>, ctx: RouterContext) => Promise<SkillResponse>;
//   getState?: () => Promise<Record<string, string>>;
//   init?: () => Promise<void>;
// }
//
// interface SkillAction {
//   name: string;
//   description: string;
//   params?: SkillParam[];
// }
//
// interface SkillResponse {
//   success: boolean;
//   voice?: string;
//   sound?: string;
//   notification?: string;
// }

export const accommodatorSkill: Skill = {
  name: "accommodator",
  description:
    "Mood-responsive audio environment. Plays locally-stored mood playlists " +
    "based on the user's detected emotional state (mood, energy, focus). " +
    "Uses the ISO principle from music therapy: first matches current mood, " +
    "then gradually steers toward a better state. " +
    "Activate when the user wants ambient/mood music, wants to feel a certain way, " +
    "or asks for help focusing or relaxing. " +
    "Also handles direct music requests like 'play some music' or 'stop the music'.",

  actions: [
    {
      name: "activate",
      description:
        "Start mood-responsive audio. Begins playing music matched to " +
        "the user's current emotional state, then gradually steers toward " +
        "a positive/neutral target.",
      params: [
        {
          name: "target",
          description:
            'Optional target mood: "calm", "uplift", "focus", "energize". ' +
            "If omitted, defaults to neutral-positive.",
          required: false,
        },
      ],
    },
    {
      name: "deactivate",
      description: "Stop mood-responsive audio and fade out playback.",
    },
    {
      name: "set_target",
      description:
        "Change the target mood state while the Accommodator is active.",
      params: [
        {
          name: "target",
          description:
            'Desired mood: "calm", "uplift", "focus", "energize", "release".',
          required: true,
        },
      ],
    },
  ],

  hints: [
    /\b(play|start|put\s+on)\s+(some\s+)?(music|something|ambient|background)\b/i,
    /\b(stop|pause|turn\s+off)\s+(the\s+)?(music|audio|sound)\b/i,
    /\b(help\s+me|want\s+to|need\s+to)\s+(focus|relax|calm\s+down|energize|feel\s+better)\b/i,
    /\b(mood|feeling|vibe)\b/i,
    /\bplay\s+(me\s+)?some\b/i,
  ],

  async init() {
    // Load playlist metadata from data/playlists/
    // Verify at least one quadrant has tracks
    // Initialize AudioEngine
    console.log("  🎭 accommodator: initialized");
  },

  async getState() {
    // Return current Accommodator state for dashboard/router
    return {
      status: "inactive", // "inactive" | "matching" | "steering" | "arrived"
      quadrant: "none",
      track: "none",
      target: "neutral-positive",
    };
  },

  async handle(
    action: string,
    params: Record<string, string>,
    ctx: RouterContext
  ): Promise<SkillResponse> {
    switch (action) {
      case "activate":
        // 1. Read current intent vector snapshot (mood, energy, taskFocus)
        // 2. Select initial quadrant via selectQuadrant()
        // 3. Start playback from that quadrant's playlist
        // 4. Begin ISO state machine (MATCHING phase)
        // 5. Start background loop watching intent vector for transitions
        return {
          success: true,
          voice: "Starting mood-responsive audio.",
          sound: "Pop",
        };

      case "deactivate":
        // 1. Fade out current track over 3 seconds
        // 2. Stop ISO state machine
        // 3. Set status to "inactive"
        return {
          success: true,
          voice: "Stopping mood audio.",
          sound: "Tink",
        };

      case "set_target":
        // 1. Parse target param to quadrant
        // 2. Update ISO target
        // 3. If already steering, recalculate path
        const target = params.target || "calm";
        return {
          success: true,
          voice: `Setting mood target to ${target}.`,
        };

      default:
        return { success: false, voice: `Unknown accommodator action: ${action}` };
    }
  },
};
```

#### Registration

In `src/listen/skills/index.ts`, add the Accommodator to `DEFAULT_SKILLS`:

```typescript
import { accommodatorSkill } from "./accommodator";

export const DEFAULT_SKILLS: Skill[] = [wellbeingSkill, accommodatorSkill];
// NOTE: musicSkill removed from DEFAULT_SKILLS (no longer a classifier target)
// It remains importable for legacy/override use.
```

In `src/listen/index.ts`, skills are registered in `initSystems()`:

```typescript
// This code already exists — no changes needed here.
// DEFAULT_SKILLS is iterated and each skill is registered:
const registry = new SkillRegistry();
for (const skill of DEFAULT_SKILLS) {
  await registry.register(skill);  // calls skill.init() if defined
}
```

#### Intent Vector Subscription

The Accommodator needs to watch intent vector snapshots. Since the Accommodator runs in-process, it can subscribe directly to updates rather than using SSE. Add a callback mechanism to `IntentVectorStore`:

```typescript
// In intent-vector.ts, add:
type IntentVectorListener = (snapshot: IntentSnapshot) => void;

class IntentVectorStore {
  private listeners: IntentVectorListener[] = [];

  /** Subscribe to intent vector updates. */
  onUpdate(fn: IntentVectorListener): void {
    this.listeners.push(fn);
  }

  /** Called within update() after computing new snapshot. */
  private notifyListeners(snapshot: IntentSnapshot): void {
    for (const fn of this.listeners) {
      try { fn(snapshot); } catch (e) { /* log and continue */ }
    }
  }
}
```

The Accommodator subscribes during `init()`:

```typescript
// In accommodator.ts init():
intentVectorStore.onUpdate((snapshot) => {
  if (this.status === "inactive") return;

  const mood = snapshot.dimensions.mood ?? 0;
  const energy = snapshot.dimensions.energy ?? 0;
  const taskFocus = snapshot.dimensions.taskFocus ?? 0;

  const selection = selectQuadrant(mood, energy, taskFocus);
  this.handleQuadrantChange(selection);
});
```

### Audio Playback Engine

**File**: `src/listen/audio.ts`

The engine uses macOS `afplay` (built-in, zero dependencies) for audio playback, with `ffplay` as a fallback. Crossfading is achieved by overlapping two `afplay` processes with volume ramps.

```typescript
/**
 * AudioEngine — local audio playback with crossfade.
 *
 * Uses macOS `afplay` (or `ffplay` fallback) via Bun.spawn.
 * Manages a queue of tracks and handles crossfading between them.
 */

import { Subprocess } from "bun";
import { readdir } from "fs/promises";
import { join } from "path";

interface PlaybackState {
  /** Currently playing track file path */
  currentTrack: string | null;
  /** Current afplay/ffplay subprocess */
  currentProcess: Subprocess | null;
  /** Volume level [0, 1] */
  volume: number;
  /** Is playback active? */
  playing: boolean;
  /** Queue of upcoming track paths */
  queue: string[];
  /** Index in current playlist */
  queueIndex: number;
}

export class AudioEngine {
  private state: PlaybackState = {
    currentTrack: null,
    currentProcess: null,
    volume: 0.5,
    playing: false,
    queue: [],
    queueIndex: 0,
  };

  /** Backend preference: "afplay" (macOS) or "ffplay" (cross-platform) */
  private backend: "afplay" | "ffplay" = "afplay";

  /**
   * Load a playlist directory into the queue.
   * Reads all .opus/.mp3 files, shuffles them.
   */
  async loadPlaylist(playlistDir: string): Promise<void> {
    const files = await readdir(playlistDir);
    const audioFiles = files
      .filter(f => f.endsWith(".opus") || f.endsWith(".mp3"))
      .filter(f => !f.startsWith("_"))  // skip _meta.json
      .map(f => join(playlistDir, f));

    // Shuffle using Fisher-Yates
    for (let i = audioFiles.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [audioFiles[i], audioFiles[j]] = [audioFiles[j], audioFiles[i]];
    }

    this.state.queue = audioFiles;
    this.state.queueIndex = 0;
  }

  /** Start playback from the current queue position. */
  async play(): Promise<void> {
    if (this.state.queue.length === 0) return;

    const track = this.state.queue[this.state.queueIndex];
    await this.playFile(track);
    this.state.playing = true;
  }

  /** Play a single audio file via afplay/ffplay. */
  private async playFile(filePath: string): Promise<void> {
    // Kill current process if any
    await this.stopCurrent();

    const volPercent = Math.round(this.state.volume * 100);

    if (this.backend === "afplay") {
      // afplay volume: 0.0 to 1.0 (linear)
      this.state.currentProcess = Bun.spawn(
        ["afplay", "-v", String(this.state.volume), filePath],
        { stdout: "ignore", stderr: "ignore" }
      );
    } else {
      // ffplay: -volume 0-100, -nodisp (no video window), -autoexit
      this.state.currentProcess = Bun.spawn(
        ["ffplay", "-nodisp", "-autoexit", "-volume", String(volPercent), filePath],
        { stdout: "ignore", stderr: "ignore" }
      );
    }

    this.state.currentTrack = filePath;

    // Wait for track to finish, then auto-advance
    this.state.currentProcess.exited.then(() => {
      if (this.state.playing) {
        this.state.queueIndex = (this.state.queueIndex + 1) % this.state.queue.length;
        this.playFile(this.state.queue[this.state.queueIndex]);
      }
    });
  }

  /**
   * Crossfade to a new playlist.
   * Overlaps the end of the current track with the start of the new one,
   * ramping volumes over `durationMs`.
   */
  async crossfadeTo(newPlaylistDir: string, durationMs: number = 3000): Promise<void> {
    const oldProcess = this.state.currentProcess;
    const targetVolume = this.state.volume;

    // Load new playlist
    await this.loadPlaylist(newPlaylistDir);
    if (this.state.queue.length === 0) return;

    // Start new track at zero volume
    const newTrack = this.state.queue[0];
    const newProcess = Bun.spawn(
      ["afplay", "-v", "0", newTrack],
      { stdout: "ignore", stderr: "ignore" }
    );

    // Ramp volumes over durationMs in steps
    const steps = 10;
    const stepMs = durationMs / steps;

    for (let i = 1; i <= steps; i++) {
      await new Promise(r => setTimeout(r, stepMs));
      // Note: afplay doesn't support runtime volume changes.
      // For true crossfade, we'd need to use ffplay or a different approach.
      // Practical implementation: kill old after delay, start new at target volume.
    }

    // Kill old process
    if (oldProcess) {
      oldProcess.kill();
    }

    // The new process is now the current one
    this.state.currentProcess = newProcess;
    this.state.currentTrack = newTrack;
  }

  /** Set volume [0, 1]. Applies to next track (afplay can't change mid-play). */
  setVolume(vol: number): void {
    this.state.volume = Math.max(0, Math.min(1, vol));
  }

  /** Stop playback and kill subprocess. */
  async stop(): Promise<void> {
    this.state.playing = false;
    await this.stopCurrent();
  }

  /** Fade out over durationMs, then stop. */
  async fadeOut(durationMs: number = 3000): Promise<void> {
    // For afplay: can't change volume mid-play, so just stop after delay
    // For a production implementation, use ffplay or sox for real fade
    this.state.playing = false;
    await new Promise(r => setTimeout(r, durationMs));
    await this.stopCurrent();
  }

  /** Get current playback info. */
  getState(): { track: string | null; playing: boolean; volume: number; queueLength: number } {
    return {
      track: this.state.currentTrack,
      playing: this.state.playing,
      volume: this.state.volume,
      queueLength: this.state.queue.length,
    };
  }

  private async stopCurrent(): Promise<void> {
    if (this.state.currentProcess) {
      this.state.currentProcess.kill();
      this.state.currentProcess = null;
    }
    this.state.currentTrack = null;
  }
}
```

**Implementation notes:**

- `afplay` is macOS-only but zero-dependency. It cannot change volume mid-playback, so "crossfade" in v1 is implemented as stop-old/start-new with a brief overlap period. True crossfading requires `sox` or `ffplay` with filter graphs.
- For v1, the "crossfade" is acceptable as a brief gap or overlap — the UX priority is that music changes happen, not that they're seamless.
- Volume based on `energy` dimension: low energy → quieter (floor 0.2), high energy → louder (ceiling 0.8).

### Classifier Changes

#### Remove `music.*` Tool Definitions

In the multi-tool classifier's system prompt (generated by `experts/experts/generate_multitool.py` and served by `experts/experts/serve_multitool.py`), remove all `music.*` tool definitions:

**Remove these from the tool list:**
```
music.play       — Start playing music or a specific song/playlist
music.pause      — Pause currently playing music
music.resume     — Resume paused music
music.skip       — Skip to next track
music.previous   — Go back to previous track
music.volume_up  — Increase volume
music.volume_down — Decrease volume
```

**Add these:**
```
accommodator.activate    — Start mood-responsive audio environment (plays music matched to user's emotional state)
accommodator.deactivate  — Stop mood-responsive audio and fade out
accommodator.set_target  — Set a target mood ("calm", "uplift", "focus", "energize", "release")
```

**Keep unchanged:**
```
wellbeing.check_in — First-person negative self-talk, burnout, self-doubt, imposter syndrome
```

#### Update VALID_TOOLS in `serve_multitool.py`

```python
VALID_TOOLS = {
    "wellbeing.check_in",
    "accommodator.activate",
    "accommodator.deactivate",
    "accommodator.set_target",
}
```

#### Retrain the Classifier

1. **Remap existing music training data**: Transform `music.play` → `accommodator.activate`, `music.pause` → `accommodator.deactivate`, `music.skip`/`music.volume_up` etc. → remove (these become internal AudioEngine operations, not classifier targets).
2. **Add new training examples** for `accommodator.set_target`:
   - "I want to feel calm" → `accommodator.set_target(target="calm")`
   - "help me get energized" → `accommodator.set_target(target="energize")`
   - "I need to focus" → `accommodator.activate(target="focus")`
   - "play something relaxing" → `accommodator.activate(target="calm")`
3. **Add negative examples**: Talking about mood/music without requesting action should not trigger.
4. **Regenerate + retrain**: `uv run generate_multitool.py && uv run train_multitool.py`
5. **Re-evaluate**: `uv run evaluate_multitool.py` — update the 64-case eval set.

#### Intent Mapping for Training Data

```
Old Tool Call              → New Tool Call
music.play                → accommodator.activate
music.play(song="X")      → accommodator.activate (or accommodator.set_target)
music.pause               → accommodator.deactivate
music.resume              → accommodator.activate
music.skip                → (remove — internal to AudioEngine)
music.previous            → (remove — internal to AudioEngine)
music.volume_up           → (remove — volume driven by energy dimension)
music.volume_down         → (remove — volume driven by energy dimension)
```

### Dashboard Integration

Add a new "Accommodator" panel to the existing SSE-driven dashboard (`src/listen/dashboard.ts`).

#### SSE Event

Emit an `"accommodator"` SSE event alongside the existing `"intentVector"` event:

```typescript
interface AccommodatorSSEPayload {
  status: "inactive" | "matching" | "steering" | "arrived";
  currentQuadrant: string;
  targetQuadrant: string;
  currentTrack: {
    title: string;
    artist: string;
    filename: string;
  } | null;
  isoProgress: {
    phase: "matching" | "steering";
    stepIndex: number;
    totalSteps: number;
    phaseElapsedMs: number;
    phaseDurationMs: number;
  } | null;
  volume: number;
}
```

#### Dashboard Panel (HTML/Canvas)

The dashboard already renders dynamically from SSE events. Add an "Accommodator" tab with:

1. **Quadrant indicator**: 2x2 grid showing the four circumplex quadrants, with the current quadrant highlighted and the target indicated with a border/arrow.
2. **Currently playing**: Track title + artist (scrolling if long).
3. **ISO progress bar**: Visual indicator of matching → steering progression. Shows current phase, elapsed time, and estimated time to next transition.
4. **Volume indicator**: Bar showing current volume level (derived from energy dimension).
5. **Status badge**: INACTIVE / MATCHING / STEERING / ARRIVED with color coding.

## Data Requirements

### Suggested Playlists per Quadrant

Each quadrant needs **20-30 tracks** (roughly 60-90 minutes of audio) to avoid repetition. Here are concrete YouTube search queries and curated playlist suggestions:

#### Uplift (High Energy + Positive Mood)

**YouTube search queries for yt-dlp:**
```
ytsearch10:upbeat happy instrumental music
ytsearch10:feel good energy boost playlist
ytsearch10:uplifting motivational instrumental
ytsearch10:happy electronic dance music instrumental
```

**Curated playlist URLs (example — verify before download):**
- YouTube Mix: "Upbeat Instrumental Music for Productivity"
- YouTube Mix: "Happy Background Music No Copyright"
- Search: "positive energy instrumental music 2024"

**Characteristics:** 110-140 BPM, major keys, bright timbres, driving rhythm

#### Release (High Energy + Negative Mood)

**YouTube search queries:**
```
ytsearch10:intense cathartic instrumental music
ytsearch10:epic dramatic orchestral music
ytsearch10:powerful emotional rock instrumental
ytsearch10:dark energetic electronic music
```

**Characteristics:** 120-160 BPM, minor keys, heavy bass, building intensity, cathartic release patterns

#### Calm (Low Energy + Positive Mood)

**YouTube search queries:**
```
ytsearch10:peaceful ambient music
ytsearch10:gentle acoustic guitar instrumental relaxing
ytsearch10:nature sounds with soft music
ytsearch10:calm piano music peaceful
```

**Characteristics:** 60-80 BPM, major/modal keys, soft dynamics, nature textures, gentle melodies

#### Comfort (Low Energy + Negative Mood)

**YouTube search queries:**
```
ytsearch10:warm soothing comfort music
ytsearch10:gentle healing music sad comfort
ytsearch10:soft melancholy piano beautiful
ytsearch10:emotional ambient comfort music
```

**Characteristics:** 50-70 BPM, minor/modal keys, warm timbres (cello, piano), gentle, acknowledging sadness without amplifying it

#### Focus (High taskFocus)

**YouTube search queries:**
```
ytsearch10:lofi hip hop beats study
ytsearch10:focus music minimal ambient
ytsearch10:deep work concentration music
ytsearch10:binaural beats focus study
```

**Characteristics:** 70-90 BPM, minimal variation, lo-fi aesthetic, no vocals, steady rhythm, low cognitive load

#### Neutral (No Strong Signal)

**YouTube search queries:**
```
ytsearch10:ambient background music subtle
ytsearch10:soft background instrumental neutral
ytsearch10:gentle white noise music blend
ytsearch10:minimal ambient textures
```

**Characteristics:** Very low presence, drone-like, barely noticeable, designed to fill silence without drawing attention

### Playlist Config File

**File**: `data/playlists/config.yaml`

```yaml
# Mood Accommodator — Playlist Sources
# Each quadrant lists YouTube URLs and/or search queries.
# Run: bun run scripts/download-playlists.ts

uplift:
  description: "Upbeat, energetic, happy — for lifting mood with high energy"
  urls: []  # Add specific YouTube video/playlist URLs here
  queries:
    - "ytsearch10:upbeat happy instrumental music"
    - "ytsearch10:feel good energy boost instrumental"
    - "ytsearch5:uplifting motivational background music"

release:
  description: "Intense, cathartic, driving — for channeling negative energy"
  urls: []
  queries:
    - "ytsearch10:intense cathartic instrumental music"
    - "ytsearch10:epic dramatic orchestral music"
    - "ytsearch5:powerful emotional cinematic music"

calm:
  description: "Ambient, peaceful, content — for maintaining positive low-energy state"
  urls: []
  queries:
    - "ytsearch10:peaceful ambient music nature"
    - "ytsearch10:gentle acoustic guitar instrumental relaxing"
    - "ytsearch5:calm piano music peaceful meditation"

comfort:
  description: "Gentle, warm, soothing — for acknowledging and easing sadness"
  urls: []
  queries:
    - "ytsearch10:warm soothing comfort music"
    - "ytsearch10:gentle healing ambient music"
    - "ytsearch5:soft melancholy piano beautiful"

focus:
  description: "Lo-fi, minimal, steady — for deep work and concentration"
  urls: []
  queries:
    - "ytsearch10:lofi hip hop beats study"
    - "ytsearch10:focus music minimal ambient work"
    - "ytsearch5:deep concentration music instrumental"

neutral:
  description: "Subtle ambient — barely noticeable background fill"
  urls: []
  queries:
    - "ytsearch10:ambient background music subtle minimal"
    - "ytsearch5:soft drone ambient textures"
```

## Configuration

### Accommodator Config

Add to `src/listen/config.ts` or create a separate config:

```typescript
export interface AccommodatorConfig {
  /** Enable/disable the Accommodator skill entirely. */
  enabled: boolean;

  /** Path to playlist data directory. */
  playlistDir: string;

  /** ISO principle: duration of matching phase (ms). */
  isoMatchDurationMs: number;

  /** ISO principle: duration of each steering step (ms). */
  isoSteerStepMs: number;

  /** Mood quadrant boundary: values below this absolute mood are "neutral". */
  neutralMoodThreshold: number;

  /** Energy threshold for high/low split. */
  energySplitThreshold: number;

  /** taskFocus threshold for override to "focus" playlist. */
  focusOverrideThreshold: number;

  /** Volume floor (never go below this). */
  volumeFloor: number;

  /** Volume ceiling (never go above this). */
  volumeCeiling: number;

  /** Audio backend: "afplay" (macOS) or "ffplay" (cross-platform). */
  audioBackend: "afplay" | "ffplay";

  /** Crossfade duration between tracks/playlists (ms). */
  crossfadeDurationMs: number;

  /** Auto-activate on session start (vs. require explicit activation). */
  autoActivate: boolean;
}

export const DEFAULT_ACCOMMODATOR_CONFIG: AccommodatorConfig = {
  enabled: true,
  playlistDir: "data/playlists",
  isoMatchDurationMs: 5 * 60_000,      // 5 minutes
  isoSteerStepMs: 10 * 60_000,          // 10 minutes per step
  neutralMoodThreshold: 0.15,
  energySplitThreshold: 0.5,
  focusOverrideThreshold: 0.6,
  volumeFloor: 0.15,
  volumeCeiling: 0.8,
  audioBackend: "afplay",
  crossfadeDurationMs: 3_000,
  autoActivate: false,
};
```

### Volume Mapping

Volume is derived from the `energy` dimension:

```typescript
function energyToVolume(energy: number, config: AccommodatorConfig): number {
  // Map energy [0, 1] to volume [floor, ceiling]
  // Low energy → quiet background
  // High energy → more presence
  const range = config.volumeCeiling - config.volumeFloor;
  return config.volumeFloor + energy * range;
}
```

## Testing Strategy

### Unit Tests

1. **`selectQuadrant()`**: Test all quadrant boundaries, neutral zone, focus override.
2. **ISO transition path**: Verify adjacency (never skip quadrants), test all starting quadrants.
3. **Volume mapping**: Test floor/ceiling clamping, energy-to-volume curve.
4. **Playlist loader**: Test file scanning, metadata parsing, shuffle.

### Integration Tests

1. **Inject test scenario** (extend `scripts/inject-test.ts` with `--scenario accommodator`):
   - Sequence of transcripts that shift mood from negative to positive
   - Verify quadrant transitions follow ISO principle
   - Verify audio engine receives correct playlist switches

2. **Classifier retrain evaluation**:
   - Remap existing 64-case eval set to new tool names
   - Verify no accuracy regression
   - Verify "not good enough" no longer triggers spurious `music.skip` (since `music.skip` is gone)

3. **End-to-end smoke test**:
   - Start pipeline with Accommodator enabled
   - Say "help me focus" → verify Accommodator activates with focus playlist
   - Say "stop the music" → verify Accommodator deactivates
   - Run for 20 minutes with varying transcripts → verify ISO transitions

### Manual Testing

1. Download at least 5 tracks per quadrant
2. Run `bun listen --moonshine` and activate Accommodator
3. Verify audio plays, crossfades happen, volume responds to energy
4. Verify dashboard shows correct state

## Rollout Plan

### Phase 1: Playlist Infrastructure (1-2 days)

- [ ] Create `data/playlists/config.yaml` with initial search queries
- [ ] Implement `scripts/download-playlists.ts` (yt-dlp wrapper)
- [ ] Download initial set of tracks (5-10 per quadrant for testing)
- [ ] Implement `_meta.json` generation from `.info.json` sidecars

### Phase 2: Audio Engine (1 day)

- [ ] Implement `src/listen/audio.ts` (AudioEngine class)
- [ ] Test playback with `afplay` backend
- [ ] Test queue management and auto-advance
- [ ] Implement basic crossfade (stop-old/start-new with overlap)

### Phase 3: Accommodator Skill (2-3 days)

- [ ] Implement `src/listen/skills/accommodator.ts`
- [ ] Implement `selectQuadrant()` with circumplex mapping
- [ ] Implement ISO state machine (MATCHING → STEERING → ARRIVED)
- [ ] Add `IntentVectorStore.onUpdate()` listener mechanism
- [ ] Wire Accommodator to intent vector updates
- [ ] Register in `DEFAULT_SKILLS`, remove `musicSkill`

### Phase 4: Classifier Retrain (1 day)

- [ ] Remap training data (music.* → accommodator.*)
- [ ] Add new training examples for `set_target`
- [ ] Regenerate, retrain, evaluate
- [ ] Update `VALID_TOOLS` in `serve_multitool.py`
- [ ] Verify no accuracy regression

### Phase 5: Dashboard + Polish (1 day)

- [ ] Add Accommodator panel to dashboard
- [ ] SSE event emission for Accommodator state
- [ ] End-to-end testing with full playlist set
- [ ] Download full playlist set (20-30 tracks per quadrant)

### Phase 6: Config + Tuning (ongoing)

- [ ] Tune ISO transition timings based on real use
- [ ] Tune quadrant boundaries based on NRC VAD calibration
- [ ] Add user-editable config (CLI flags or config file)

## Future: AI Music Generation

For v1, pre-downloaded playlists are simpler and more reliable. But the groundwork supports future replacement with on-the-fly generation:

### MusicGen-Small via MLX

- **Model**: MusicGen-Small (300M params, ~3GB)
- **Runtime**: MLX (Apple Silicon native)
- **Weights**: `jasonvassallo/musicgen-small-mlx` on HuggingFace (March 2026, brand-new MLX port)
- **Capability**: Text-conditioned music generation ("calm ambient piano in C major, 70 BPM")
- **Latency**: ~5-10 seconds for 10 seconds of audio on M-series chips

### Integration Path

1. Replace playlist-based track selection with generation prompts per quadrant
2. Generate 30-60 second clips on-demand, cache to filesystem
3. Crossfade between generated clips
4. ISO principle still applies — the generation prompt changes, not just the playlist

### Prompt Templates (Future)

```
uplift:  "upbeat happy instrumental music, major key, 120 BPM, bright synths and acoustic guitar"
release: "intense dramatic orchestral music, minor key, 140 BPM, building crescendo"
calm:    "peaceful ambient music, soft piano and nature sounds, 70 BPM, gentle and spacious"
comfort: "warm soothing music, gentle cello and piano, 60 BPM, melancholy but hopeful"
focus:   "lo-fi hip hop beats, minimal, steady rhythm, 80 BPM, vinyl crackle, no vocals"
neutral: "subtle ambient drone, barely perceptible, soft textures, very quiet"
```

This is deferred because:
- Pre-downloaded playlists have zero latency
- Real music sounds better than current generation models
- No GPU contention with the classifier model
- Simpler debugging (known audio files vs. generated)

## Open Questions

1. **Legal considerations**: Downloading from YouTube via yt-dlp for personal/private use is a gray area. The playlists should use Creative Commons or royalty-free music where possible. Consider using Free Music Archive, ccMixter, or Incompetech instead.

2. **afplay crossfade limitations**: `afplay` cannot change volume mid-playback. Should we require `sox` (which supports fade effects) as a dependency, or is stop-overlap-start acceptable for v1?

3. **Accommodator activation UX**: Should the Accommodator auto-activate when mood drops below a threshold (with prior consent), or always require explicit activation? The current design requires explicit opt-in, but auto-activation with a wellbeing gate could be valuable.

4. **Existing Apple Music integration**: The current `music.ts` skill controls Apple Music via AppleScript/MusicKit. Should direct music commands like "play Bohemian Rhapsody" still route to Apple Music (as an override), or should ALL music requests go through the Accommodator? Recommendation: Keep `music.ts` importable but not registered as a classifier target. If the user says a specific song name, the Accommodator could detect this via the `song` param and delegate to `music.ts` internally.

5. **Multiple audio sources**: If the user is already playing Apple Music and activates the Accommodator, what happens? Should the Accommodator pause Apple Music first? Or coexist? Recommendation: Pause Apple Music via AppleScript when Accommodator activates, resume when it deactivates.

6. **Wellbeing integration**: When the wellbeing gate fires (detected distress), should the Accommodator auto-shift to "comfort" playlist regardless of ISO state? This could be a powerful integration point — the wellbeing skill and Accommodator working together.

7. **BPM detection**: Is automated BPM detection worth the complexity for v1? It would enable smoother ISO transitions (match BPM between outgoing and incoming tracks). Could use `aubio` or `librosa` in a post-download processing step. Recommendation: Defer to v2.

8. **Track repeat avoidance**: With 20-30 tracks per quadrant, users will hear repeats within a few hours. Should we implement a "recently played" buffer that avoids replaying tracks within a configurable window? Recommendation: Yes, simple set of last-N filenames to exclude from shuffle.
