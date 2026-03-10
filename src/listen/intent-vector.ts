/**
 * IntentVector — real-time intent signal engine.
 *
 * Maintains a multi-dimensional activation vector that decays over time,
 * updated by classifier (expert) results. Each dimension represents a
 * different intent signal (music, wellbeing, engagement, taskFocus).
 *
 * Decay uses exponential half-life per dimension so stale signals fade
 * naturally. A ring buffer of snapshots provides trend computation and
 * short-term history for downstream consumers.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single expert/classifier result from the classify pipeline. */
export interface ExpertResult {
  skill: string;
  match: boolean;
  confidence?: number;
}

/** Dimension keys tracked by the vector. */
export type DimensionKey = "music" | "wellbeing" | "engagement" | "taskFocus";

/** Per-dimension numeric map. */
export type DimensionMap = Record<DimensionKey, number>;

/** Point-in-time snapshot of the full intent vector. */
export interface IntentVectorSnapshot {
  timestamp: string; // ISO string
  dimensions: {
    music: number; //    [0, 1] — from classifier match
    wellbeing: number; // [0, 1] — from classifier match
    engagement: number; // [0, 1] — chunk density (chunks/min)
    taskFocus: number; //  [0, 1] — ratio of skill-matched to total chunks
  };
  /** Per-dimension trend: positive = rising, negative = falling, ~0 = stable. */
  trends: {
    music: number;
    wellbeing: number;
    engagement: number;
    taskFocus: number;
  };
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DIMENSION_KEYS: readonly DimensionKey[] = [
  "music",
  "wellbeing",
  "engagement",
  "taskFocus",
] as const;

/** Half-life per dimension (ms). Controls how fast each signal decays. */
const HALF_LIVES: Readonly<DimensionMap> = {
  music: 45_000,
  wellbeing: 120_000,
  engagement: 60_000,
  taskFocus: 30_000,
};

/** Baseline each dimension decays toward. */
const BASELINES: Readonly<DimensionMap> = {
  music: 0,
  wellbeing: 0,
  engagement: 0,
  taskFocus: 0,
};

/** Max snapshots retained (~5 min at 1 snapshot/sec). */
const MAX_SNAPSHOTS = 300;

/** Window for engagement chunk-rate computation (ms). */
const ENGAGEMENT_WINDOW_MS = 60_000;

/** Max expected chunks per minute for speech (normalization factor). */
const MAX_CHUNKS_PER_MIN = 12;

/** Window for taskFocus skill-match ratio (ms). */
const TASK_FOCUS_WINDOW_MS = 30_000;

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
  private lastUpdated: Record<DimensionKey, number>;

  // Ring buffer of snapshots.
  private ring: IntentVectorSnapshot[] = [];
  private ringHead = 0; // next write position
  private ringSize = 0; // number of valid entries

  // Rolling windows for engagement & taskFocus.
  private chunkTimestamps: number[] = [];
  private chunkMatchLog: { ts: number; matched: boolean }[] = [];

  constructor() {
    this.dims = { music: 0, wellbeing: 0, engagement: 0, taskFocus: 0 };
    this.lastUpdated = {
      music: 0,
      wellbeing: 0,
      engagement: 0,
      taskFocus: 0,
    };
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Ingest a batch of expert/classifier results and advance the vector.
   *
   * @param expertResults  Results from the classify pipeline.
   * @param now            Optional epoch ms (defaults to Date.now()); useful for testing.
   * @returns The newly computed snapshot.
   */
  update(
    expertResults: ExpertResult[],
    now: number = Date.now(),
  ): IntentVectorSnapshot {
    // 1. Decay all dimensions based on time elapsed since last update.
    for (const key of DIMENSION_KEYS) {
      const elapsed = this.lastUpdated[key] > 0 ? now - this.lastUpdated[key] : 0;
      if (elapsed > 0) {
        this.dims[key] = decay(
          this.dims[key],
          BASELINES[key],
          HALF_LIVES[key],
          elapsed,
        );
      }
    }

    // 2. Apply expert matches — only raise, never lower an already-high activation.
    const anyMatch = expertResults.some((r) => r.match);
    for (const result of expertResults) {
      if (!result.match) continue;
      const key = result.skill as DimensionKey;
      if (!DIMENSION_KEYS.includes(key)) continue;
      const activation = result.confidence ?? 0.95;
      this.dims[key] = Math.max(this.dims[key], clamp01(activation));
      this.lastUpdated[key] = now;
    }

    // 3. Record chunk arrival for engagement / taskFocus rolling windows.
    this.chunkTimestamps.push(now);
    this.chunkMatchLog.push({ ts: now, matched: anyMatch });

    // Evict stale entries outside the largest window we care about.
    const evictBefore = now - Math.max(ENGAGEMENT_WINDOW_MS, TASK_FOCUS_WINDOW_MS);
    this.chunkTimestamps = this.chunkTimestamps.filter((t) => t >= evictBefore);
    this.chunkMatchLog = this.chunkMatchLog.filter((e) => e.ts >= evictBefore);

    // 4. Compute engagement: chunks in last 60s / MAX_CHUNKS_PER_MIN.
    const engagementCutoff = now - ENGAGEMENT_WINDOW_MS;
    const chunksInWindow = this.chunkTimestamps.filter((t) => t >= engagementCutoff).length;
    this.dims.engagement = clamp01(chunksInWindow / MAX_CHUNKS_PER_MIN);
    this.lastUpdated.engagement = now;

    // 5. Compute taskFocus: matched / total in last 30s window.
    const focusCutoff = now - TASK_FOCUS_WINDOW_MS;
    const focusEntries = this.chunkMatchLog.filter((e) => e.ts >= focusCutoff);
    if (focusEntries.length > 0) {
      const matched = focusEntries.filter((e) => e.matched).length;
      this.dims.taskFocus = clamp01(matched / focusEntries.length);
    }
    this.lastUpdated.taskFocus = now;

    // 6. Compute trends by comparing to ~10s-ago snapshot.
    const trends = this.computeTrends(now);

    // 7. Build snapshot and push to ring buffer.
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
    for (const key of DIMENSION_KEYS) {
      this.dims[key] = 0;
      this.lastUpdated[key] = 0;
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
    const trends: DimensionMap = { music: 0, wellbeing: 0, engagement: 0, taskFocus: 0 };
    const past = this.findSnapshotNear(now - TREND_LOOKBACK_MS);
    if (!past) return trends;

    for (const key of DIMENSION_KEYS) {
      const diff = this.dims[key] - past.dimensions[key];
      trends[key] = clamp(diff, -1, 1);
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
// Helpers
// ---------------------------------------------------------------------------

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
