/**
 * AudioEngine — local audio playback with crossfade.
 *
 * Uses macOS `afplay` (or `ffplay` fallback) via Bun.spawn.
 * Manages a queue of tracks and handles crossfading between playlists.
 *
 * Design decisions:
 *   - afplay is macOS-only but zero-dependency (built-in)
 *   - afplay cannot change volume mid-playback, so "crossfade" in v1
 *     is implemented as stop-old/start-new with a brief overlap
 *   - Volume is mapped from the energy intent dimension:
 *     low energy → quiet (floor 0.15), high energy → louder (ceiling 0.8)
 *   - Tracks auto-advance when finished, wrapping around the shuffled queue
 *   - Fisher-Yates shuffle for playlist randomization
 */

import { readdir } from "fs/promises";
import { join, basename } from "path";

// ── Types ──────────────────────────────────────────────────────────

export interface PlaybackState {
  /** Currently playing track file path */
  currentTrack: string | null;
  /** Current afplay/ffplay subprocess */
  currentProcess: import("bun").Subprocess | null;
  /** Volume level [0, 1] */
  volume: number;
  /** Is playback active? */
  playing: boolean;
  /** Queue of upcoming track paths */
  queue: string[];
  /** Index in current playlist */
  queueIndex: number;
  /** Name of the currently loaded playlist (quadrant name) */
  playlistName: string | null;
}

export interface AudioEngineConfig {
  /** Audio backend: "afplay" (macOS) or "ffplay" (cross-platform). */
  backend: "afplay" | "ffplay";
  /** Volume floor (never go below this). */
  volumeFloor: number;
  /** Volume ceiling (never go above this). */
  volumeCeiling: number;
  /** Crossfade duration between tracks/playlists (ms). */
  crossfadeDurationMs: number;
}

export const DEFAULT_AUDIO_CONFIG: AudioEngineConfig = {
  backend: "afplay",
  volumeFloor: 0.15,
  volumeCeiling: 0.8,
  crossfadeDurationMs: 3_000,
};

/** Public snapshot of playback state (safe to serialize). */
export interface PlaybackInfo {
  track: string | null;
  trackName: string | null;
  playing: boolean;
  volume: number;
  queueLength: number;
  playlistName: string | null;
}

// ── AudioEngine ────────────────────────────────────────────────────

export class AudioEngine {
  private state: PlaybackState = {
    currentTrack: null,
    currentProcess: null,
    volume: 0.5,
    playing: false,
    queue: [],
    queueIndex: 0,
    playlistName: null,
  };

  private config: AudioEngineConfig;

  /** Recently played track paths — used to avoid repeats within a session. */
  private recentlyPlayed: string[] = [];
  private readonly recentlyPlayedMax = 30;

  constructor(config?: Partial<AudioEngineConfig>) {
    this.config = { ...DEFAULT_AUDIO_CONFIG, ...config };
  }

  // ── Playlist Management ────────────────────────────────────────

  /**
   * Load a playlist directory into the queue.
   * Reads all .opus/.mp3/.m4a files, shuffles them (Fisher-Yates),
   * and avoids recently-played tracks.
   */
  async loadPlaylist(playlistDir: string, name?: string): Promise<number> {
    let files: string[];
    try {
      files = await readdir(playlistDir);
    } catch {
      console.warn(`  ⚠ audio: playlist dir not found: ${playlistDir}`);
      return 0;
    }

    let audioFiles = files
      .filter(f => /\.(opus|mp3|m4a)$/i.test(f))
      .filter(f => !f.startsWith("_"))
      .map(f => join(playlistDir, f));

    // Filter out recently played tracks (if we have enough alternatives)
    const recentSet = new Set(this.recentlyPlayed);
    const fresh = audioFiles.filter(f => !recentSet.has(f));
    if (fresh.length >= 3) {
      audioFiles = fresh;
    }

    // Shuffle using Fisher-Yates
    for (let i = audioFiles.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [audioFiles[i], audioFiles[j]] = [audioFiles[j], audioFiles[i]];
    }

    this.state.queue = audioFiles;
    this.state.queueIndex = 0;
    this.state.playlistName = name || basename(playlistDir);

    return audioFiles.length;
  }

  // ── Playback Control ───────────────────────────────────────────

  /** Start playback from the current queue position. */
  async play(): Promise<boolean> {
    if (this.state.queue.length === 0) {
      console.warn("  ⚠ audio: no tracks in queue");
      return false;
    }

    const track = this.state.queue[this.state.queueIndex];
    this.state.playing = true;
    await this.playFile(track);
    return true;
  }

  /** Stop playback and kill subprocess. */
  async stop(): Promise<void> {
    this.state.playing = false;
    await this.killCurrent();
  }

  /** Skip to the next track in the queue. */
  async skip(): Promise<void> {
    if (!this.state.playing || this.state.queue.length === 0) return;
    this.state.queueIndex = (this.state.queueIndex + 1) % this.state.queue.length;
    await this.playFile(this.state.queue[this.state.queueIndex]);
  }

  /**
   * Fade out over durationMs, then stop.
   * Note: afplay can't change volume mid-play, so this is a timed stop.
   * For production, use ffplay or sox for real fade effects.
   */
  async fadeOut(durationMs?: number): Promise<void> {
    const ms = durationMs ?? this.config.crossfadeDurationMs;
    this.state.playing = false;
    await new Promise(r => setTimeout(r, ms));
    await this.killCurrent();
  }

  // ── Crossfade ──────────────────────────────────────────────────

  /**
   * Crossfade to a new playlist.
   *
   * In v1 (afplay backend), this is a stop-old/start-new with a brief
   * overlap. True crossfading would require ffplay with filter graphs
   * or sox, which is deferred to v2.
   *
   * Returns the number of tracks in the new playlist (0 = failed).
   */
  async crossfadeTo(
    newPlaylistDir: string,
    name?: string,
    durationMs?: number,
  ): Promise<number> {
    const ms = durationMs ?? this.config.crossfadeDurationMs;
    const wasPlaying = this.state.playing;

    // Load new playlist
    const trackCount = await this.loadPlaylist(newPlaylistDir, name);
    if (trackCount === 0) return 0;

    if (wasPlaying && this.state.currentProcess) {
      // Brief overlap: start new track, then kill old after a delay
      const oldProcess = this.state.currentProcess;
      const newTrack = this.state.queue[0];

      // Start new track
      this.state.playing = true;
      await this.spawnPlayer(newTrack);

      // Kill old after half the crossfade duration
      setTimeout(() => {
        try { oldProcess.kill(); } catch { /* already exited */ }
      }, ms / 2);
    } else if (wasPlaying) {
      // No current process but was playing — just start new
      this.state.playing = true;
      await this.playFile(this.state.queue[0]);
    }

    return trackCount;
  }

  // ── Volume ─────────────────────────────────────────────────────

  /**
   * Set volume [0, 1]. Clamped to [floor, ceiling].
   * Takes effect on the next track (afplay can't change mid-play).
   */
  setVolume(vol: number): void {
    this.state.volume = Math.max(
      this.config.volumeFloor,
      Math.min(this.config.volumeCeiling, vol),
    );
  }

  /**
   * Map energy dimension [0, 1] to volume [floor, ceiling].
   * Low energy → quiet background. High energy → more presence.
   */
  setVolumeFromEnergy(energy: number): void {
    const range = this.config.volumeCeiling - this.config.volumeFloor;
    this.setVolume(this.config.volumeFloor + energy * range);
  }

  // ── State ──────────────────────────────────────────────────────

  /** Get current playback info (safe to serialize). */
  getInfo(): PlaybackInfo {
    return {
      track: this.state.currentTrack,
      trackName: this.state.currentTrack
        ? basename(this.state.currentTrack).replace(/\.\w+$/, "")
        : null,
      playing: this.state.playing,
      volume: this.state.volume,
      queueLength: this.state.queue.length,
      playlistName: this.state.playlistName,
    };
  }

  /** Check if the engine is actively playing. */
  get isPlaying(): boolean {
    return this.state.playing;
  }

  /** Get the current playlist name. */
  get currentPlaylist(): string | null {
    return this.state.playlistName;
  }

  // ── Internals ──────────────────────────────────────────────────

  /** Play a single audio file via afplay/ffplay. */
  private async playFile(filePath: string): Promise<void> {
    await this.killCurrent();
    await this.spawnPlayer(filePath);
  }

  /** Spawn the audio player subprocess. */
  private async spawnPlayer(filePath: string): Promise<void> {
    this.state.currentTrack = filePath;

    // Track recently played
    this.recentlyPlayed.push(filePath);
    if (this.recentlyPlayed.length > this.recentlyPlayedMax) {
      this.recentlyPlayed.shift();
    }

    if (this.config.backend === "afplay") {
      this.state.currentProcess = Bun.spawn(
        ["afplay", "-v", String(this.state.volume), filePath],
        { stdout: "ignore", stderr: "ignore" },
      );
    } else {
      const volPercent = Math.round(this.state.volume * 100);
      this.state.currentProcess = Bun.spawn(
        ["ffplay", "-nodisp", "-autoexit", "-volume", String(volPercent), filePath],
        { stdout: "ignore", stderr: "ignore" },
      );
    }

    // Auto-advance when track finishes — only if this is still the current process
    // (prevents race condition during crossfade where old process exit kills new track)
    const proc = this.state.currentProcess;
    proc.exited.then(() => {
      if (this.state.currentProcess === proc && this.state.playing && this.state.queue.length > 0) {
        this.state.queueIndex = (this.state.queueIndex + 1) % this.state.queue.length;
        this.playFile(this.state.queue[this.state.queueIndex]);
      }
    });
  }

  /** Kill the current player subprocess. */
  private async killCurrent(): Promise<void> {
    if (this.state.currentProcess) {
      try {
        this.state.currentProcess.kill();
      } catch {
        /* already exited */
      }
      this.state.currentProcess = null;
    }
    this.state.currentTrack = null;
  }
}
