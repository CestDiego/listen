/**
 * SessionStore — structured timeline of everything that happens.
 *
 * Captures transcripts, gate results, watchlist matches, analysis,
 * and user corrections in a single timeline. Persists to JSON file
 * and pushes updates via SSE to the dashboard.
 */

import { writeFile, mkdir } from "fs/promises";
import { dirname } from "path";
import type { GateResult, AnalysisResult } from "./config";
import type { WatchlistMatch } from "./watchlist";
import type { IntentVectorSnapshot, GateResult as ActivationGateResult } from "./intent-vector";

// ── Types ──────────────────────────────────────────────────────────

export interface TimelineEntry {
  id: string;
  cycle: number;
  timestamp: string;
  /** Original whisper transcription */
  original: string;
  /** User-corrected text (null = not corrected) */
  corrected: string | null;
  /** Best available text (corrected ?? original) */
  text: string;
  durationSeconds: number;
  events: TimelineEvent[];
}

export type TimelineEvent =
  | { type: "gate"; score: number; reason: string; latencyMs: number }
  | { type: "gate.escalation"; score: number; reason: string; latencyMs: number }
  | { type: "watchlist"; patternId: string; category: string; severity: string; trigger: string }
  | { type: "router"; interest: number; reason: string; skills: string[]; latencyMs: number }
  | { type: "skill"; skill: string; action: string; success: boolean; voice?: string }
  | { type: "analysis"; insights: string; triggerReason: string }
  | { type: "silence" }
  | { type: "correction"; from: string; to: string; at: string };

// ── Router Decision (full observability) ──────────────────────────

/** A matched skill with full details for the decision log. */
export interface DecisionSkillMatch {
  skill: string;
  action: string;
  params: Record<string, string>;
  confidence: number;
  /** Was the skill actually executed (false if on cooldown)? */
  executed?: boolean;
  /** Execution result */
  success?: boolean;
  /** Voice response if any */
  voice?: string;
}

/** Per-expert classification snapshot for full observability. */
export interface DecisionExpertResult {
  skill: string;
  match: boolean;
  action?: string;
  confidence?: number;
  /** Expert-side inference latency (ms) */
  expertMs: number;
  /** Full round-trip latency including HTTP (ms) */
  roundTripMs: number;
  /** Classification status */
  status: "ok" | "error" | "timeout" | "too_short";
  /** Error detail if any */
  error?: string;
}

/** Full snapshot of a classification decision — the core observability unit. */
export interface RouterDecision {
  /** Unique decision id */
  id: string;
  /** Timeline entry this decision belongs to */
  entryId: string;
  /** When the decision was made */
  timestamp: string;
  /** The transcript that was classified */
  transcript: string;
  /** Rolling buffer context sent to the classifier */
  bufferContext: string;
  /** Skill state at time of classification (e.g. "music: status=playing") */
  skillState: string;
  /** Recent skill executions the classifier saw */
  recentSkills: Array<{
    skill: string;
    action: string;
    success: boolean;
    voice?: string;
    agoSeconds: number;
  }>;
  /** Interest score 0-10 */
  interest: number;
  /** Brief reasoning */
  reason: string;
  /** All skill matches with full details */
  matches: DecisionSkillMatch[];
  /** Total wall-clock time for parallel classification (ms) */
  classifyMs: number;
  /** Sum of individual expert round-trip times (ms) — shows parallelism gain */
  expertSumMs: number;
  /** Per-expert classification results */
  expertResults: DecisionExpertResult[];
  /** Whether interest exceeded escalation threshold */
  escalated: boolean;
  /** Word count at time of classification */
  wordCount: number;

  // ── Deprecated: kept for backward compat ──
  /** @deprecated Use classifyMs instead */
  latencyMs: number;
}

export interface Session {
  id: string;
  startedAt: string;
  config: Record<string, unknown>;
  timeline: TimelineEntry[];
  /** Full router decision log for observability */
  decisions: RouterDecision[];
}

// ── SSE subscriber type ────────────────────────────────────────────

type SSESubscriber = (event: string, data: unknown) => void;

// ── Store ──────────────────────────────────────────────────────────

export class SessionStore {
  private session: Session;
  private savePath: string;
  private saveTimer: ReturnType<typeof setTimeout> | null = null;
  private subscribers: Set<SSESubscriber> = new Set();
  private intentVector: IntentVectorSnapshot | null = null;
  private intentVectorHistory: IntentVectorSnapshot[] = [];

  constructor(savePath: string, config: Record<string, unknown>) {
    this.savePath = savePath;
    this.session = {
      id: `session-${Date.now()}`,
      startedAt: new Date().toISOString(),
      config,
      timeline: [],
      decisions: [],
    };
  }

  /** Subscribe to real-time updates (for SSE). */
  subscribe(fn: SSESubscriber): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  private notify(event: string, data: unknown) {
    for (const fn of this.subscribers) {
      try { fn(event, data); } catch {}
    }
  }

  /** Add a new transcript chunk to the timeline. */
  addChunk(cycle: number, original: string, durationSeconds: number): TimelineEntry {
    const entry: TimelineEntry = {
      id: `chunk-${String(cycle).padStart(5, "0")}`,
      cycle,
      timestamp: new Date().toISOString(),
      original,
      corrected: null,
      text: original,
      durationSeconds,
      events: original ? [] : [{ type: "silence" }],
    };

    this.session.timeline.push(entry);
    this.notify("chunk", entry);
    this.debounceSave();
    return entry;
  }

  /** Add a gate result to the latest (or specified) entry. */
  addGate(result: GateResult, latencyMs: number, escalated: boolean, entryId?: string) {
    const entry = entryId
      ? this.session.timeline.find((e) => e.id === entryId)
      : this.lastNonSilent();
    if (!entry) return;

    const evt: TimelineEvent = escalated
      ? { type: "gate.escalation", score: result.score, reason: result.reason, latencyMs }
      : { type: "gate", score: result.score, reason: result.reason, latencyMs };

    entry.events.push(evt);
    this.notify("gate", { entryId: entry.id, ...evt });
    this.debounceSave();
  }

  /** Add a watchlist match to the latest entry. */
  addWatchlistMatch(match: WatchlistMatch, entryId?: string) {
    const entry = entryId
      ? this.session.timeline.find((e) => e.id === entryId)
      : this.lastNonSilent();
    if (!entry) return;

    const evt: TimelineEvent = {
      type: "watchlist",
      patternId: match.pattern.id,
      category: match.pattern.category,
      severity: match.pattern.severity,
      trigger: match.trigger,
    };

    entry.events.push(evt);
    this.notify("watchlist", { entryId: entry.id, ...evt });
    this.debounceSave();
  }

  /** Add a router result to the latest entry (legacy slim method). */
  addRouterResult(
    interest: number,
    reason: string,
    skills: string[],
    latencyMs: number,
    entryId?: string
  ) {
    const entry = entryId
      ? this.session.timeline.find((e) => e.id === entryId)
      : this.lastNonSilent();
    if (!entry) return;

    const evt: TimelineEvent = {
      type: "router",
      interest,
      reason,
      skills,
      latencyMs,
    };

    entry.events.push(evt);
    this.notify("router", { entryId: entry.id, ...evt });
    this.debounceSave();
  }

  /**
   * Add a full router decision — the primary observability record.
   * This captures everything the router saw, decided, and what happened.
   */
  addRouterDecision(decision: Omit<RouterDecision, "id">): RouterDecision {
    const id = `decision-${String(this.session.decisions.length + 1).padStart(5, "0")}`;
    const full: RouterDecision = { id, ...decision };
    this.session.decisions.push(full);

    // Also add timeline event (for inline rendering)
    const entry = this.session.timeline.find((e) => e.id === decision.entryId);
    if (entry) {
      const evt: TimelineEvent = {
        type: "router",
        interest: decision.interest,
        reason: decision.reason,
        skills: decision.matches.map((m) => `${m.skill}.${m.action}`),
        latencyMs: decision.latencyMs,
      };
      entry.events.push(evt);
      // Notify timeline subscribers so the dashboard shows inline router badge
      this.notify("router", { entryId: entry.id, ...evt });
    }

    // Push the full decision via SSE for dashboard observability
    this.notify("decision", full);
    this.debounceSave();
    return full;
  }

  /**
   * Update a decision's skill match with execution results.
   * Called after each skill handler completes.
   */
  updateDecisionSkill(
    decisionId: string,
    skill: string,
    action: string,
    executed: boolean,
    success: boolean,
    voice?: string
  ) {
    const decision = this.session.decisions.find((d) => d.id === decisionId);
    if (!decision) return;

    const match = decision.matches.find(
      (m) => m.skill === skill && m.action === action
    );
    if (match) {
      match.executed = executed;
      match.success = success;
      match.voice = voice;
    } else {
      console.warn(`  ⚠ updateDecisionSkill: no match for ${skill}.${action} in ${decisionId}`);
    }

    // Push update via SSE
    this.notify("decision_update", {
      decisionId,
      skill,
      action,
      executed,
      success,
      voice,
    });
    this.debounceSave();
  }

  /** Add a skill execution result to the latest entry. */
  addSkillResult(
    skill: string,
    action: string,
    success: boolean,
    voice?: string,
    entryId?: string
  ) {
    const entry = entryId
      ? this.session.timeline.find((e) => e.id === entryId)
      : this.lastNonSilent();
    if (!entry) return;

    const evt: TimelineEvent = {
      type: "skill",
      skill,
      action,
      success,
      voice,
    };

    entry.events.push(evt);
    this.notify("skill", { entryId: entry.id, ...evt });
    this.debounceSave();
  }

  /** Add analysis results to the latest entry. */
  addAnalysis(result: AnalysisResult, entryId?: string) {
    const entry = entryId
      ? this.session.timeline.find((e) => e.id === entryId)
      : this.lastNonSilent();
    if (!entry) return;

    const evt: TimelineEvent = {
      type: "analysis",
      insights: result.insights,
      triggerReason: result.triggerReason,
    };

    entry.events.push(evt);
    this.notify("analysis", { entryId: entry.id, ...evt });
    this.debounceSave();
  }

  /** Store and broadcast an intent vector snapshot with optional gate result. */
  emitIntentVector(snapshot: IntentVectorSnapshot, history: IntentVectorSnapshot[], gate?: ActivationGateResult): void {
    this.intentVector = snapshot;
    this.intentVectorHistory = history;
    this.notify("intentVector", { snapshot, history, gate: gate ?? null });
  }

  /** Correct a transcription. Returns true if found. */
  correct(entryId: string, correctedText: string): boolean {
    const entry = this.session.timeline.find((e) => e.id === entryId);
    if (!entry) return false;

    const evt: TimelineEvent = {
      type: "correction",
      from: entry.text,
      to: correctedText,
      at: new Date().toISOString(),
    };

    entry.corrected = correctedText;
    entry.text = correctedText;
    entry.events.push(evt);
    this.notify("correction", { entryId: entry.id, ...evt });
    this.debounceSave();
    return true;
  }

  /** Get the full session data (for API / dashboard). */
  getSession(): Session & { intentVector?: IntentVectorSnapshot | null; intentVectorHistory?: IntentVectorSnapshot[] } {
    return {
      ...this.session,
      intentVector: this.intentVector,
      intentVectorHistory: this.intentVectorHistory,
    };
  }

  /** Get a slice of the timeline. */
  getTimeline(offset = 0, limit = 100): TimelineEntry[] {
    return this.session.timeline.slice(offset, offset + limit);
  }

  /** Get all router decisions (for /api/decisions). */
  getDecisions(): RouterDecision[] {
    return this.session.decisions;
  }

  /** Get stats — now includes router decision metrics. */
  getStats() {
    const entries = this.session.timeline;
    const decisions = this.session.decisions;
    const nonSilent = entries.filter((e) => e.original);
    const corrections = entries.filter((e) => e.corrected !== null);
    const watchlistHits = entries.filter((e) =>
      e.events.some((ev) => ev.type === "watchlist")
    );
    const escalations = entries.filter((e) =>
      e.events.some((ev) => ev.type === "gate.escalation")
    );

    // Classification decision stats
    const skillActivations = decisions.reduce(
      (sum, d) => sum + d.matches.filter((m) => m.executed).length,
      0
    );
    const avgClassifyMs = decisions.length > 0
      ? Math.round(decisions.reduce((sum, d) => sum + d.classifyMs, 0) / decisions.length)
      : 0;
    const avgExpertSumMs = decisions.length > 0
      ? Math.round(decisions.reduce((sum, d) => sum + d.expertSumMs, 0) / decisions.length)
      : 0;
    const avgInterest = decisions.length > 0
      ? Number((decisions.reduce((sum, d) => sum + d.interest, 0) / decisions.length).toFixed(1))
      : 0;
    const avgParallelGain = avgClassifyMs > 0
      ? Number((avgExpertSumMs / avgClassifyMs).toFixed(2))
      : 1;

    return {
      totalChunks: entries.length,
      transcribedChunks: nonSilent.length,
      corrections: corrections.length,
      watchlistHits: watchlistHits.length,
      escalations: escalations.length,
      totalWords: nonSilent.reduce(
        (sum, e) => sum + (e.text.trim() ? e.text.trim().split(/\s+/).length : 0),
        0
      ),
      // Classification observability stats
      totalDecisions: decisions.length,
      skillActivations,
      avgClassifyMs,
      avgExpertSumMs,
      avgParallelGain,
      avgInterest,
      escalationRate: decisions.length > 0
        ? Number((decisions.filter((d) => d.escalated).length / decisions.length * 100).toFixed(1))
        : 0,
      intentVector: this.intentVector,
    };
  }

  // ── Persistence ──────────────────────────────────────────────────

  private lastNonSilent(): TimelineEntry | undefined {
    for (let i = this.session.timeline.length - 1; i >= 0; i--) {
      if (this.session.timeline[i].original) return this.session.timeline[i];
    }
    return undefined;
  }

  private debounceSave() {
    if (this.saveTimer) clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => this.save(), 2000);
  }

  async save(): Promise<void> {
    try {
      await mkdir(dirname(this.savePath), { recursive: true });
      await writeFile(this.savePath, JSON.stringify(this.session, null, 2), "utf-8");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  ⚠ session save failed: ${msg}`);
    }
  }
}
