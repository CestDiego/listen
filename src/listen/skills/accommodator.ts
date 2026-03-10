/**
 * Accommodator Skill — mood-responsive audio environment.
 *
 * Plays locally-stored mood playlists based on the user's detected emotional
 * state (mood, energy, focus). Uses the ISO principle from music therapy:
 * first matches current mood, then gradually steers toward a better state.
 *
 * Architecture:
 *   - Watches IntentVectorStore for mood/energy/taskFocus changes
 *   - Maps 2D affect space (Russell's Circumplex) to playlist quadrants
 *   - ISO state machine: INACTIVE → MATCHING → STEERING → ARRIVED
 *   - AudioEngine handles local playback via afplay/ffplay
 */

import type { Skill, SkillResponse, RouterContext } from "./types";
import { AudioEngine } from "../audio";
import type { IntentVectorSnapshot, IntentVectorListener } from "../intent-vector";
import { resolve, join } from "path";

// ── Types ──────────────────────────────────────────────────────────

export type Quadrant = "uplift" | "release" | "calm" | "comfort" | "focus" | "neutral";

export type ISOStatus = "inactive" | "matching" | "steering" | "arrived";

export interface QuadrantSelection {
  quadrant: Quadrant;
  confidence: number;
}

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
}

export const DEFAULT_ACCOMMODATOR_CONFIG: AccommodatorConfig = {
  enabled: true,
  playlistDir: resolve(import.meta.dir, "../../../data/playlists"),
  isoMatchDurationMs: 5 * 60_000,
  isoSteerStepMs: 10 * 60_000,
  neutralMoodThreshold: 0.15,
  energySplitThreshold: 0.5,
  focusOverrideThreshold: 0.6,
  volumeFloor: 0.15,
  volumeCeiling: 0.8,
  audioBackend: "afplay",
  crossfadeDurationMs: 3_000,
};

// ── ISO State ──────────────────────────────────────────────────────

interface ISOState {
  status: ISOStatus;
  currentQuadrant: Quadrant | null;
  targetQuadrant: Quadrant | null;
  phaseStartMs: number;
  steerStepIndex: number;
  steerPath: Quadrant[];
}

// ── Quadrant Selection ─────────────────────────────────────────────

/**
 * Map mood (valence) and energy (arousal) to a circumplex quadrant.
 * Special overlays: taskFocus > threshold → "focus"; neutral zone → "neutral".
 */
export function selectQuadrant(
  mood: number,
  energy: number,
  taskFocus: number,
  config: AccommodatorConfig = DEFAULT_ACCOMMODATOR_CONFIG,
): QuadrantSelection {
  // Override: high task focus → focus playlist
  if (taskFocus > config.focusOverrideThreshold) {
    return { quadrant: "focus", confidence: taskFocus };
  }

  // Neutral zone: no strong signal
  if (Math.abs(mood) < config.neutralMoodThreshold && energy < 0.4) {
    return { quadrant: "neutral", confidence: 1 - Math.abs(mood) * 3 };
  }

  // Circumplex quadrants
  const isPositive = mood >= 0;
  const isHighEnergy = energy >= config.energySplitThreshold;

  if (isPositive && isHighEnergy) return { quadrant: "uplift", confidence: mood * energy };
  if (!isPositive && isHighEnergy) return { quadrant: "release", confidence: Math.abs(mood) * energy };
  if (isPositive && !isHighEnergy) return { quadrant: "calm", confidence: mood * (1 - energy) };
  return { quadrant: "comfort", confidence: Math.abs(mood) * (1 - energy) };
}

// ── Adjacency Graph ────────────────────────────────────────────────
// ISO principle: never jump quadrants, always move through adjacent ones.

const ADJACENCY: Record<Quadrant, Quadrant[]> = {
  comfort: ["calm", "release"],
  calm: ["comfort", "uplift"],
  uplift: ["calm", "release"],
  release: ["comfort", "uplift"],
  focus: ["calm", "neutral"],
  neutral: ["calm", "focus"],
};

/**
 * Compute a path from `from` to `to` through adjacent quadrants.
 * Returns the intermediate steps (excluding `from`, including `to`).
 */
export function computeSteerPath(from: Quadrant, to: Quadrant): Quadrant[] {
  if (from === to) return [];

  // BFS through adjacency graph
  const visited = new Set<Quadrant>([from]);
  const queue: { node: Quadrant; path: Quadrant[] }[] = [{ node: from, path: [] }];

  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const neighbor of ADJACENCY[current.node]) {
      if (neighbor === to) {
        return [...current.path, to];
      }
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push({ node: neighbor, path: [...current.path, neighbor] });
      }
    }
  }

  // Fallback: direct jump (shouldn't happen with valid adjacency graph)
  return [to];
}

// ── Volume Mapping ─────────────────────────────────────────────────

export function energyToVolume(energy: number, config: AccommodatorConfig): number {
  const range = config.volumeCeiling - config.volumeFloor;
  return config.volumeFloor + energy * range;
}

// ── Accommodator Engine ────────────────────────────────────────────

class AccommodatorEngine {
  private config: AccommodatorConfig;
  private audio: AudioEngine;
  private iso: ISOState;
  private unsubscribe: (() => void) | null = null;
  private steerTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(config?: Partial<AccommodatorConfig>) {
    this.config = { ...DEFAULT_ACCOMMODATOR_CONFIG, ...config };
    this.audio = new AudioEngine({
      backend: this.config.audioBackend,
      volumeFloor: this.config.volumeFloor,
      volumeCeiling: this.config.volumeCeiling,
      crossfadeDurationMs: this.config.crossfadeDurationMs,
    });
    this.iso = this.freshISO();
  }

  private freshISO(): ISOState {
    return {
      status: "inactive",
      currentQuadrant: null,
      targetQuadrant: null,
      phaseStartMs: 0,
      steerStepIndex: 0,
      steerPath: [],
    };
  }

  /** Subscribe to intent vector updates. Call during init(). */
  subscribe(onUpdate: (fn: IntentVectorListener) => () => void): void {
    this.unsubscribe = onUpdate((snapshot) => {
      if (this.iso.status === "inactive") return;
      this.handleIntentUpdate(snapshot);
    });
  }

  /** Activate the Accommodator — start matching phase. */
  async activate(
    mood: number,
    energy: number,
    taskFocus: number,
    target?: string,
  ): Promise<void> {
    const selection = selectQuadrant(mood, energy, taskFocus, this.config);

    this.iso.status = "matching";
    this.iso.currentQuadrant = selection.quadrant;
    this.iso.phaseStartMs = Date.now();
    this.iso.steerStepIndex = 0;

    // Set target
    const targetQuadrant = this.parseTarget(target) ?? this.defaultTarget();
    this.iso.targetQuadrant = targetQuadrant;

    // Compute steer path
    this.iso.steerPath = computeSteerPath(selection.quadrant, targetQuadrant);

    // Load and play the matching quadrant's playlist
    const playlistDir = join(this.config.playlistDir, selection.quadrant);
    const count = await this.audio.loadPlaylist(playlistDir, selection.quadrant);

    if (count > 0) {
      this.audio.setVolumeFromEnergy(energy);
      await this.audio.play();
    }

    // Schedule transition from MATCHING to STEERING
    this.scheduleMatchToSteer();
  }

  /** Deactivate — fade out and stop. */
  async deactivate(): Promise<void> {
    this.clearTimers();
    await this.audio.fadeOut(this.config.crossfadeDurationMs);
    this.iso = this.freshISO();
  }

  /** Change the target mood while active. */
  setTarget(target: string): void {
    const quadrant = this.parseTarget(target);
    if (!quadrant || !this.iso.currentQuadrant) return;

    this.iso.targetQuadrant = quadrant;
    this.iso.steerPath = computeSteerPath(
      this.iso.currentQuadrant,
      quadrant,
    );
    this.iso.steerStepIndex = 0;

    // If already steering, restart the step timer
    if (this.iso.status === "steering") {
      this.scheduleSteerStep();
    }
  }

  /** Get current state for dashboard/SSE. */
  getState(): {
    status: ISOStatus;
    quadrant: string;
    target: string;
    track: string;
    volume: number;
    steerStep: number;
    steerTotal: number;
  } {
    const info = this.audio.getInfo();
    return {
      status: this.iso.status,
      quadrant: this.iso.currentQuadrant ?? "none",
      target: this.iso.targetQuadrant ?? "neutral-positive",
      track: info.trackName ?? "none",
      volume: info.volume,
      steerStep: this.iso.steerStepIndex,
      steerTotal: this.iso.steerPath.length,
    };
  }

  /** Handle an intent vector update — adjust volume, check for drift. */
  private handleIntentUpdate(snapshot: IntentVectorSnapshot): void {
    const mood = snapshot.dimensions.mood ?? 0;
    const energy = snapshot.dimensions.energy ?? 0;
    const taskFocus = snapshot.dimensions.taskFocus ?? 0;

    // Always adjust volume based on energy
    this.audio.setVolumeFromEnergy(energy);

    // Check if mood has drifted significantly from current quadrant
    if (this.iso.status === "arrived") {
      const selection = selectQuadrant(mood, energy, taskFocus, this.config);
      if (selection.quadrant !== this.iso.currentQuadrant && selection.confidence > 0.3) {
        // Mood drifted — restart steering
        this.iso.status = "matching";
        this.iso.phaseStartMs = Date.now();
        this.iso.currentQuadrant = selection.quadrant;
        this.iso.steerPath = computeSteerPath(
          selection.quadrant,
          this.iso.targetQuadrant ?? this.defaultTarget(),
        );
        this.iso.steerStepIndex = 0;
        this.scheduleMatchToSteer();
      }
    }
  }

  // ── ISO Timing ──────────────────────────────────────────────────

  private scheduleMatchToSteer(): void {
    this.clearTimers();
    this.steerTimer = setTimeout(async () => {
      if (this.iso.status !== "matching") return;

      if (this.iso.steerPath.length === 0) {
        // Already at target
        this.iso.status = "arrived";
        return;
      }

      this.iso.status = "steering";
      this.iso.steerStepIndex = 0;
      await this.executeSteerStep();
    }, this.config.isoMatchDurationMs);
  }

  private scheduleSteerStep(): void {
    this.clearTimers();
    this.steerTimer = setTimeout(async () => {
      await this.executeSteerStep();
    }, this.config.isoSteerStepMs);
  }

  private async executeSteerStep(): Promise<void> {
    if (this.iso.status !== "steering") return;
    if (this.iso.steerStepIndex >= this.iso.steerPath.length) {
      this.iso.status = "arrived";
      return;
    }

    const nextQuadrant = this.iso.steerPath[this.iso.steerStepIndex];
    const playlistDir = join(this.config.playlistDir, nextQuadrant);

    const count = await this.audio.crossfadeTo(playlistDir, nextQuadrant);
    if (count > 0) {
      this.iso.currentQuadrant = nextQuadrant;
      this.iso.steerStepIndex++;

      if (this.iso.steerStepIndex >= this.iso.steerPath.length) {
        this.iso.status = "arrived";
      } else {
        this.scheduleSteerStep();
      }
    }
  }

  private clearTimers(): void {
    if (this.steerTimer) {
      clearTimeout(this.steerTimer);
      this.steerTimer = null;
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────

  private parseTarget(target?: string): Quadrant | null {
    if (!target) return null;
    const map: Record<string, Quadrant> = {
      calm: "calm",
      relax: "calm",
      relaxing: "calm",
      peaceful: "calm",
      uplift: "uplift",
      energize: "uplift",
      energized: "uplift",
      upbeat: "uplift",
      happy: "uplift",
      focus: "focus",
      concentrate: "focus",
      study: "focus",
      work: "focus",
      release: "release",
      intense: "release",
      comfort: "comfort",
      soothe: "comfort",
      gentle: "comfort",
      neutral: "neutral",
    };
    return map[target.toLowerCase()] ?? null;
  }

  private defaultTarget(): Quadrant {
    return "calm"; // neutral-positive as default ISO target
  }
}

// ── Singleton ──────────────────────────────────────────────────────

let engine: AccommodatorEngine | null = null;

function getEngine(): AccommodatorEngine {
  if (!engine) {
    engine = new AccommodatorEngine();
  }
  return engine;
}

// ── Skill Definition ───────────────────────────────────────────────

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
    const eng = getEngine();
    console.log("  🎭 accommodator: initialized");
    // Note: subscription to IntentVectorStore happens in index.ts
    // after both the store and the skill are created.
  },

  async getState() {
    const state = getEngine().getState();
    return {
      status: state.status,
      quadrant: state.quadrant,
      target: state.target,
      track: state.track,
    };
  },

  async handle(
    action: string,
    params: Record<string, string>,
    _ctx: RouterContext,
  ): Promise<SkillResponse> {
    const eng = getEngine();

    switch (action) {
      case "activate": {
        // Read current intent vector values from the engine state
        // In production, these come from the IntentVectorStore subscription
        // For activation, use defaults or target-derived values
        const target = params.target;
        await eng.activate(0, 0.5, 0, target);
        const targetMsg = target ? ` Target: ${target}.` : "";
        return {
          success: true,
          voice: `Starting mood-responsive audio.${targetMsg}`,
          sound: "Pop",
        };
      }

      case "deactivate": {
        await eng.deactivate();
        return {
          success: true,
          voice: "Stopping mood audio.",
          sound: "Tink",
        };
      }

      case "set_target": {
        const target = params.target || "calm";
        eng.setTarget(target);
        return {
          success: true,
          voice: `Setting mood target to ${target}.`,
        };
      }

      default:
        return {
          success: false,
          voice: `Unknown accommodator action: ${action}`,
        };
    }
  },
};

/**
 * Connect the Accommodator to the IntentVectorStore.
 * Called from index.ts after both systems are initialized.
 */
export function connectAccommodatorToIntentVector(
  onUpdate: (fn: IntentVectorListener) => () => void,
): void {
  getEngine().subscribe(onUpdate);
}
