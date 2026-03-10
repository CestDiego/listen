/**
 * IntentVector — real-time intent signal engine.
 *
 * Maintains a multi-dimensional activation vector that decays over time,
 * updated by classifier (expert) results and computed heuristics.
 * Each dimension is defined in DIMENSION_DEFS — adding a new dimension
 * is a single config entry; the engine, gate, and dashboard adapt automatically.
 *
 * Mood and energy dimensions use the NRC VAD Lexicon (44,728 words) for
 * word-level valence/arousal scoring instead of naive heuristics.
 *
 * Decay uses exponential half-life per dimension so stale signals fade
 * naturally. A ring buffer of snapshots provides trend computation and
 * short-term history for downstream consumers.
 */

import { chunkVAD, isLexiconAvailable, loadLexicon } from "./nrc-vad";

// ---------------------------------------------------------------------------
// Dimension Definition — the single source of truth
// ---------------------------------------------------------------------------

/**
 * How a dimension gets its value:
 *   "classifier" — set directly from classifier match confidence
 *   "computed"   — calculated by a compute hook each update cycle
 */
export type DimensionSource = "classifier" | "computed";

/** Full definition of a single intent dimension. */
export interface DimensionDef {
  /** Unique key (used in maps, SSE payloads, dashboard). */
  key: string;
  /** Human-readable label for dashboard. */
  label: string;
  /** Short label for compact displays (radar axis). */
  shortLabel: string;
  /** CSS color (hex) for charts. */
  color: string;
  /** Half-life in ms — how fast the signal decays toward baseline. */
  halfLifeMs: number;
  /** Value the dimension decays toward. */
  baseline: number;
  /** Minimum value (for clamping). Default 0. */
  min?: number;
  /** Maximum value (for clamping). Default 1. */
  max?: number;
  /** How this dimension is sourced. */
  source: DimensionSource;
  /**
   * For "computed" dimensions: a function called each update cycle.
   * Receives the compute context and returns the new raw value.
   * The engine clamps the result to [min, max] after calling this.
   */
  compute?: (ctx: ComputeContext) => number;
  /** For computed dimensions that use a rolling window: window size in ms. */
  windowMs?: number;
}

/** Context passed to computed dimension hooks. */
export interface ComputeContext {
  /** Current epoch ms. */
  now: number;
  /** Timestamps of all chunks in the rolling window. */
  chunkTimestamps: number[];
  /** Match log entries in the rolling window. */
  chunkMatchLog: Array<{ ts: number; matched: boolean }>;
  /** The current expert results for this update cycle. */
  expertResults: ExpertResult[];
  /** The raw transcript text for this chunk (if available). */
  transcript?: string;
}

// ---------------------------------------------------------------------------
// Dimension Registry — add new dimensions here
// ---------------------------------------------------------------------------

// Eagerly load the NRC VAD Lexicon at module init (44,728 words, ~2MB, <100ms).
// If the file is missing, loadLexicon() warns and returns empty — graceful fallback.
loadLexicon();

export const DIMENSION_DEFS: readonly DimensionDef[] = [
  {
    key: "music",
    label: "Music",
    shortLabel: "music",
    color: "#58a6ff",
    halfLifeMs: 45_000,
    baseline: 0,
    source: "classifier",
  },
  {
    key: "wellbeing",
    label: "Wellbeing",
    shortLabel: "wellbeing",
    color: "#f85149",
    halfLifeMs: 120_000,
    baseline: 0,
    source: "classifier",
  },
  {
    key: "engagement",
    label: "Engagement",
    shortLabel: "engage",
    color: "#3fb950",
    halfLifeMs: 60_000,
    baseline: 0,
    source: "computed",
    windowMs: 60_000,
    compute: (ctx) => {
      // Chunks in last 60s / MAX_CHUNKS_PER_MIN
      const MAX_CHUNKS_PER_MIN = 12;
      const cutoff = ctx.now - 60_000;
      const chunksInWindow = ctx.chunkTimestamps.filter((t) => t >= cutoff).length;
      return chunksInWindow / MAX_CHUNKS_PER_MIN;
    },
  },
  {
    key: "taskFocus",
    label: "Task Focus",
    shortLabel: "focus",
    color: "#d29922",
    halfLifeMs: 30_000,
    baseline: 0,
    source: "computed",
    windowMs: 30_000,
    compute: (ctx) => {
      // Matched / total in last 30s window
      const cutoff = ctx.now - 30_000;
      const entries = ctx.chunkMatchLog.filter((e) => e.ts >= cutoff);
      if (entries.length === 0) return 0;
      const matched = entries.filter((e) => e.matched).length;
      return matched / entries.length;
    },
  },
  {
    key: "mood",
    label: "Mood",
    shortLabel: "mood",
    color: "#bc8cff",
    halfLifeMs: 120_000,
    baseline: 0,
    min: -1,
    max: 1,
    source: "computed",
    compute: (ctx) => {
      // NRC VAD Lexicon: average valence across all recognized words.
      // Range: [-1, +1]. Positive = pleasant, negative = unpleasant.
      // Falls back to 0 (neutral) if lexicon unavailable or no words recognized.
      const text = ctx.transcript ?? "";
      if (!text) return 0;
      const vad = chunkVAD(text);
      return vad.valence;
    },
  },
  {
    key: "energy",
    label: "Energy",
    shortLabel: "energy",
    color: "#f778ba",
    halfLifeMs: 60_000,
    baseline: 0,
    source: "computed",
    compute: (ctx) => {
      // Blend two signals:
      //   1. NRC VAD arousal: lexical energy (excited/calm) [-1, +1] → rescale to [0, 1]
      //   2. Speech rate: words per chunk / expected max [0, 1]
      // Final = weighted blend (0.6 arousal + 0.4 speech rate) for robustness.
      const text = ctx.transcript ?? "";
      if (!text) return 0;

      // Signal 1: NRC arousal (lexical energy)
      let arousalNorm = 0.5; // neutral if no lexicon
      if (isLexiconAvailable()) {
        const vad = chunkVAD(text);
        // Rescale [-1, +1] → [0, 1]: (arousal + 1) / 2
        arousalNorm = (vad.arousal + 1) / 2;
      }

      // Signal 2: speech rate
      const wordCount = text.split(/\s+/).filter(Boolean).length;
      const MAX_WORDS_PER_CHUNK = 30;
      const speechRate = Math.min(1, wordCount / MAX_WORDS_PER_CHUNK);

      // Blend: arousal is the primary signal, speech rate is secondary
      return 0.6 * arousalNorm + 0.4 * speechRate;
    },
  },
];

// ---------------------------------------------------------------------------
// Derived helpers — everything below is computed from DIMENSION_DEFS
// ---------------------------------------------------------------------------

/** All dimension keys as a readonly array. */
export const DIMENSION_KEYS: readonly string[] = DIMENSION_DEFS.map((d) => d.key);

/** Type-safe dimension key (union of all registered keys). */
export type DimensionKey = string;

/** Per-dimension numeric map. */
export type DimensionMap = Record<string, number>;

/** Build a DimensionMap with every key set to a given value (or per-key baseline). */
export function buildDimensionMap(fillValue?: number): DimensionMap {
  const map: DimensionMap = {};
  for (const def of DIMENSION_DEFS) {
    map[def.key] = fillValue ?? def.baseline;
  }
  return map;
}

/** Pre-built Map for O(1) dimension lookup by key. */
const DIMENSION_MAP = new Map<string, DimensionDef>(
  DIMENSION_DEFS.map((d) => [d.key, d]),
);

/** Get the DimensionDef for a key, or undefined. O(1). */
export function getDimensionDef(key: string): DimensionDef | undefined {
  return DIMENSION_MAP.get(key);
}

/** Metadata payload sent to dashboard via SSE init event. */
export interface DimensionMeta {
  key: string;
  label: string;
  shortLabel: string;
  color: string;
  min: number;
  max: number;
  source: DimensionSource;
}

/** Build the metadata array for the dashboard. */
export function buildDimensionMeta(): DimensionMeta[] {
  return DIMENSION_DEFS.map((d) => ({
    key: d.key,
    label: d.label,
    shortLabel: d.shortLabel,
    color: d.color,
    min: d.min ?? 0,
    max: d.max ?? 1,
    source: d.source,
  }));
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single expert/classifier result from the classify pipeline. */
export interface ExpertResult {
  skill: string;
  match: boolean;
  confidence?: number;
}

/** Point-in-time snapshot of the full intent vector. */
export interface IntentVectorSnapshot {
  timestamp: string; // ISO string
  dimensions: DimensionMap;
  /** Per-dimension trend: positive = rising, negative = falling, ~0 = stable. */
  trends: DimensionMap;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Max snapshots retained (~5 min at 1 snapshot/sec). */
const MAX_SNAPSHOTS = 300;

/** How far back to look for trend comparison (ms). */
const TREND_LOOKBACK_MS = 10_000;

// ---------------------------------------------------------------------------
// Decay function
// ---------------------------------------------------------------------------

/**
 * Exponential decay toward a baseline.
 *
 * After `halfLifeMs` milliseconds the distance from baseline is halved.
 *
 * @param value     Current value.
 * @param baseline  Value to decay toward.
 * @param halfLifeMs  Time for the signal to lose half its distance from baseline.
 * @param elapsedMs   Time elapsed since the value was set.
 * @returns Decayed value.
 */
export function decay(
  value: number,
  baseline: number,
  halfLifeMs: number,
  elapsedMs: number,
): number {
  return baseline + (value - baseline) * Math.pow(0.5, elapsedMs / halfLifeMs);
}

// ---------------------------------------------------------------------------
// IntentVectorStore
// ---------------------------------------------------------------------------

export class IntentVectorStore {
  // Current dimension activations.
  private dims: DimensionMap;

  // Timestamp (epoch ms) of the last update per dimension.
  private lastUpdated: Record<string, number>;

  // Ring buffer of snapshots.
  private ring: IntentVectorSnapshot[] = [];
  private ringHead = 0; // next write position
  private ringSize = 0; // number of valid entries

  // Rolling windows for computed dimensions.
  private chunkTimestamps: number[] = [];
  private chunkMatchLog: { ts: number; matched: boolean }[] = [];

  constructor() {
    this.dims = buildDimensionMap();
    this.lastUpdated = {};
    for (const def of DIMENSION_DEFS) {
      this.lastUpdated[def.key] = 0;
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Ingest a batch of expert/classifier results and advance the vector.
   *
   * @param expertResults  Results from the classify pipeline.
   * @param now            Optional epoch ms (defaults to Date.now()); useful for testing.
   * @param transcript     Optional transcript text for computed dimensions.
   * @returns The newly computed snapshot.
   */
  update(
    expertResults: ExpertResult[],
    now: number = Date.now(),
    transcript?: string,
  ): IntentVectorSnapshot {
    // 1. Decay all dimensions based on time elapsed since last update.
    for (const def of DIMENSION_DEFS) {
      const elapsed = this.lastUpdated[def.key] > 0 ? now - this.lastUpdated[def.key] : 0;
      if (elapsed > 0) {
        this.dims[def.key] = decay(
          this.dims[def.key],
          def.baseline,
          def.halfLifeMs,
          elapsed,
        );
      }
    }

    // 2. Apply expert matches — only raise, never lower an already-high activation.
    //    Only applies to "classifier" source dimensions.
    const anyMatch = expertResults.some((r) => r.match);
    for (const result of expertResults) {
      if (!result.match) continue;
      const def = getDimensionDef(result.skill);
      if (!def || def.source !== "classifier") continue;
      const activation = result.confidence ?? 0.95;
      const maxVal = def.max ?? 1;
      const minVal = def.min ?? 0;
      this.dims[def.key] = Math.max(this.dims[def.key], clamp(activation, minVal, maxVal));
      this.lastUpdated[def.key] = now;
    }

    // 3. Record chunk arrival for rolling windows.
    this.chunkTimestamps.push(now);
    this.chunkMatchLog.push({ ts: now, matched: anyMatch });

    // Evict stale entries outside the largest window we care about.
    const maxWindow = Math.max(
      ...DIMENSION_DEFS.filter((d) => d.windowMs).map((d) => d.windowMs!),
      60_000, // minimum 60s
    );
    const evictBefore = now - maxWindow;
    this.chunkTimestamps = this.chunkTimestamps.filter((t) => t >= evictBefore);
    this.chunkMatchLog = this.chunkMatchLog.filter((e) => e.ts >= evictBefore);

    // 4. Run compute hooks for "computed" dimensions.
    const computeCtx: ComputeContext = {
      now,
      chunkTimestamps: this.chunkTimestamps,
      chunkMatchLog: this.chunkMatchLog,
      expertResults,
      transcript,
    };

    for (const def of DIMENSION_DEFS) {
      if (def.source === "computed" && def.compute) {
        const raw = def.compute(computeCtx);
        const minVal = def.min ?? 0;
        const maxVal = def.max ?? 1;
        this.dims[def.key] = clamp(raw, minVal, maxVal);
        this.lastUpdated[def.key] = now;
      }
    }

    // 5. Compute trends by comparing to ~10s-ago snapshot.
    const trends = this.computeTrends(now);

    // 6. Build snapshot and push to ring buffer.
    const snap: IntentVectorSnapshot = {
      timestamp: new Date(now).toISOString(),
      dimensions: { ...this.dims },
      trends,
    };
    this.pushSnapshot(snap);

    return snap;
  }

  /** Return the current snapshot (does NOT advance or decay). */
  snapshot(): IntentVectorSnapshot {
    const trends = this.computeTrends(Date.now());
    return {
      timestamp: new Date().toISOString(),
      dimensions: { ...this.dims },
      trends,
    };
  }

  /** Return up to `maxItems` most recent snapshots (oldest first). */
  history(maxItems?: number): IntentVectorSnapshot[] {
    const all = this.ringToArray();
    if (maxItems !== undefined && maxItems < all.length) {
      return all.slice(all.length - maxItems);
    }
    return all;
  }

  /** Reset all dimensions, clear rolling windows and history. */
  reset(): void {
    for (const def of DIMENSION_DEFS) {
      this.dims[def.key] = def.baseline;
      this.lastUpdated[def.key] = 0;
    }
    this.ring = [];
    this.ringHead = 0;
    this.ringSize = 0;
    this.chunkTimestamps = [];
    this.chunkMatchLog = [];
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  /**
   * Compute per-dimension trend by comparing current dims to a snapshot
   * from ~TREND_LOOKBACK_MS ago. Returns 0 for all dims if no history.
   */
  private computeTrends(now: number): DimensionMap {
    const trends = buildDimensionMap(0);
    const past = this.findSnapshotNear(now - TREND_LOOKBACK_MS);
    if (!past) return trends;

    for (const def of DIMENSION_DEFS) {
      const diff = this.dims[def.key] - (past.dimensions[def.key] ?? 0);
      trends[def.key] = clamp(diff, -1, 1);
    }
    return trends;
  }

  /**
   * Find the snapshot closest to a target timestamp (epoch ms).
   * Returns undefined if the ring buffer is empty.
   */
  private findSnapshotNear(targetMs: number): IntentVectorSnapshot | undefined {
    if (this.ringSize === 0) return undefined;

    const snapshots = this.ringToArray();
    let best: IntentVectorSnapshot | undefined;
    let bestDelta = Infinity;

    for (const snap of snapshots) {
      const delta = Math.abs(new Date(snap.timestamp).getTime() - targetMs);
      if (delta < bestDelta) {
        bestDelta = delta;
        best = snap;
      }
    }
    return best;
  }

  /** Push a snapshot into the fixed-size ring buffer. */
  private pushSnapshot(snap: IntentVectorSnapshot): void {
    if (this.ring.length < MAX_SNAPSHOTS) {
      this.ring.push(snap);
    } else {
      this.ring[this.ringHead] = snap;
    }
    this.ringHead = (this.ringHead + 1) % MAX_SNAPSHOTS;
    if (this.ringSize < MAX_SNAPSHOTS) this.ringSize++;
  }

  /** Materialize the ring buffer into a time-ordered array. */
  private ringToArray(): IntentVectorSnapshot[] {
    if (this.ringSize < MAX_SNAPSHOTS) {
      // Buffer hasn't wrapped yet — entries 0..ringSize-1 are in order.
      return this.ring.slice(0, this.ringSize);
    }
    // Buffer has wrapped: oldest entry is at ringHead.
    return [
      ...this.ring.slice(this.ringHead),
      ...this.ring.slice(0, this.ringHead),
    ];
  }
}

// ---------------------------------------------------------------------------
// Activation Gate — post-classification routing bias
// ---------------------------------------------------------------------------

/** Gate state for a skill activation. */
export type GateState = "idle" | "vigilant" | "active";

/** Result of the activation gate check. */
export interface GateResult {
  /** Current gate state. */
  state: GateState;
  /** The target dimension key this gate monitors. */
  targetDimension: string;
  /** The decayed activation level for the target dimension [0, 1]. */
  activationLevel: number;
  /** Effective threshold being used (lower when vigilant). */
  effectiveThreshold: number;
  /** Whether the gate promoted a match this cycle. */
  promoted: boolean;
  /** If promoted, the synthetic confidence assigned. */
  promotedConfidence?: number;
  /** Time since last genuine trigger (ms), or null if never. */
  timeSinceLastTrigger: number | null;

  // ── Backward compat aliases ──
  /** @deprecated Use activationLevel */
  wellbeingLevel: number;
}

/** Configuration for the activation gate. */
export interface GateConfig {
  /** The dimension key this gate monitors. Default: "wellbeing" */
  targetDimension: string;
  /** The skill name to match/promote for. Default: same as targetDimension */
  targetSkill: string;
  /** The action to inject when promoting. Default: "check_in" */
  promotionAction: string;
  /** Classifier confidence to enter active state. Default: 0.55 */
  activateThreshold: number;
  /** Intent vector level to drop back to idle. Default: 0.15 */
  deactivateThreshold: number;
  /** Confidence assigned to promoted (synthetic) matches. Default: 0.40 */
  promotionConfidence: number;
  /** Maximum gate duration in ms (hard cap). Default: 600_000 (10 min) */
  maxGateDurationMs: number;
  /** Minimum classifier confidence to be considered a "near miss". Default: 0.0
   *  (promote any time we're vigilant, regardless of classifier score) */
  nearMissFloor: number;
}

export const DEFAULT_GATE_CONFIG: GateConfig = {
  targetDimension: "wellbeing",
  targetSkill: "wellbeing",
  promotionAction: "check_in",
  activateThreshold: 0.55,
  deactivateThreshold: 0.15,
  promotionConfidence: 0.40,
  maxGateDurationMs: 600_000,
  nearMissFloor: 0.0,
};

export class ActivationGate {
  private state: GateState = "idle";
  private lastGenuineTriggerMs: number | null = null;
  readonly config: GateConfig;

  constructor(config?: Partial<GateConfig>) {
    this.config = { ...DEFAULT_GATE_CONFIG, ...config };
  }

  /** The dimension key this gate monitors. */
  get targetDimension(): string {
    return this.config.targetDimension;
  }

  /** The skill name this gate matches/promotes. */
  get targetSkill(): string {
    return this.config.targetSkill;
  }

  /**
   * Evaluate the gate after classification.
   *
   * @param classifierMatched       Did the classifier return a match for the target skill?
   * @param classifierConfidence    Confidence of the match (0 if no match)
   * @param intentVectorLevel       Current target dimension level from IntentVectorStore
   * @param now                     Current time (epoch ms), for testing
   * @returns GateResult with state, whether to promote, etc.
   */
  evaluate(
    classifierMatched: boolean,
    classifierConfidence: number,
    intentVectorLevel: number,
    now: number = Date.now(),
  ): GateResult {
    const timeSince = this.lastGenuineTriggerMs !== null
      ? now - this.lastGenuineTriggerMs
      : null;

    // Hard cap: if we've been gated for too long, force idle
    if (timeSince !== null && timeSince > this.config.maxGateDurationMs) {
      this.state = "idle";
    }

    // State transitions
    if (classifierMatched && classifierConfidence >= this.config.activateThreshold) {
      // Genuine strong match → active, renew timer
      this.state = "active";
      this.lastGenuineTriggerMs = now;
    } else if (classifierMatched) {
      // Classifier matched but below activate threshold — still counts as genuine
      // Keep current state or upgrade to vigilant
      this.lastGenuineTriggerMs = now;
      if (this.state === "idle") {
        this.state = "vigilant";
      }
    } else if (this.state === "active") {
      // Was active, classifier didn't match this cycle → transition to vigilant
      this.state = "vigilant";
    } else if (this.state === "vigilant") {
      // Check if we should drop to idle
      if (intentVectorLevel < this.config.deactivateThreshold) {
        this.state = "idle";
      }
      // else: stay vigilant (hysteresis — between thresholds)
    }
    // idle + no classifier match → stays idle

    // Determine effective threshold
    const effectiveThreshold = this.state === "idle"
      ? this.config.activateThreshold
      : this.config.deactivateThreshold;

    // Promotion logic: only when vigilant AND classifier didn't match
    let promoted = false;
    let promotedConfidence: number | undefined;

    if (
      this.state === "vigilant" &&
      !classifierMatched &&
      intentVectorLevel >= this.config.deactivateThreshold
    ) {
      promoted = true;
      // Scale promotion confidence by the current activation level
      // Higher activation = higher confidence in the promotion
      promotedConfidence = this.config.promotionConfidence *
        Math.min(1, intentVectorLevel / this.config.activateThreshold);
    }

    return {
      state: this.state,
      targetDimension: this.config.targetDimension,
      activationLevel: intentVectorLevel,
      effectiveThreshold,
      promoted,
      promotedConfidence,
      timeSinceLastTrigger: timeSince,
      // backward compat
      wellbeingLevel: intentVectorLevel,
    };
  }

  /** Get current state without modifying it. */
  getState(): GateState {
    return this.state;
  }

  /** Reset gate to idle. */
  reset(): void {
    this.state = "idle";
    this.lastGenuineTriggerMs = null;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
