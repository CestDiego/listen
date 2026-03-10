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
  | { type: "analysis"; insights: string; triggerReason: string }
  | { type: "silence" }
  | { type: "correction"; from: string; to: string; at: string };

export interface Session {
  id: string;
  startedAt: string;
  config: Record<string, unknown>;
  timeline: TimelineEntry[];
}

// ── SSE subscriber type ────────────────────────────────────────────

type SSESubscriber = (event: string, data: unknown) => void;

// ── Store ──────────────────────────────────────────────────────────

export class SessionStore {
  private session: Session;
  private savePath: string;
  private saveTimer: ReturnType<typeof setTimeout> | null = null;
  private subscribers: Set<SSESubscriber> = new Set();

  constructor(savePath: string, config: Record<string, unknown>) {
    this.savePath = savePath;
    this.session = {
      id: `session-${Date.now()}`,
      startedAt: new Date().toISOString(),
      config,
      timeline: [],
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
  getSession(): Session {
    return this.session;
  }

  /** Get a slice of the timeline. */
  getTimeline(offset = 0, limit = 100): TimelineEntry[] {
    return this.session.timeline.slice(offset, offset + limit);
  }

  /** Get stats. */
  getStats() {
    const entries = this.session.timeline;
    const nonSilent = entries.filter((e) => e.original);
    const corrections = entries.filter((e) => e.corrected !== null);
    const watchlistHits = entries.filter((e) =>
      e.events.some((ev) => ev.type === "watchlist")
    );
    const escalations = entries.filter((e) =>
      e.events.some((ev) => ev.type === "gate.escalation")
    );

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
