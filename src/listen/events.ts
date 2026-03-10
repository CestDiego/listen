/**
 * Event emitter — writes structured events to a JSONL file.
 *
 * Other processes/agents can `tail -f` the event log to react.
 * Each line is a self-contained JSON object.
 *
 * Event types:
 *   - watchlist.match     → a watchlist pattern triggered
 *   - gate.check          → a gate check was performed
 *   - gate.escalation     → gate score exceeded threshold
 *   - analysis.complete   → full analysis was generated
 */

import { appendFile, mkdir } from "fs/promises";
import { dirname } from "path";

// ── Types ──────────────────────────────────────────────────────────

export interface ListenEvent {
  timestamp: string;
  type: string;
  [key: string]: unknown;
}

export interface WatchlistEvent extends ListenEvent {
  type: "watchlist.match";
  patternId: string;
  category: string;
  severity: string;
  trigger: string;
  matchedText: string;
  context: string;
}

export interface GateEvent extends ListenEvent {
  type: "gate.check" | "gate.escalation";
  score: number;
  reason: string;
  wordCount: number;
}

export interface AnalysisEvent extends ListenEvent {
  type: "analysis.complete";
  triggerReason: string;
  insights: string;
}

// ── Emitter ────────────────────────────────────────────────────────

export class EventEmitter {
  private logPath: string;
  private initialized = false;
  private verbose: boolean;

  constructor(logPath: string, verbose = false) {
    this.logPath = logPath;
    this.verbose = verbose;
  }

  /** Ensure the log directory exists. */
  private async init(): Promise<void> {
    if (this.initialized) return;
    await mkdir(dirname(this.logPath), { recursive: true });
    this.initialized = true;
  }

  /** Emit an event — appends to JSONL log. */
  async emit(event: ListenEvent): Promise<void> {
    await this.init();

    const line = JSON.stringify(event) + "\n";

    try {
      await appendFile(this.logPath, line, "utf-8");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  ⚠ event write failed: ${msg}`);
    }

    if (this.verbose) {
      const icon = eventIcon(event.type);
      console.log(`  ${icon} EVENT → ${event.type} ${eventSummary(event)}`);
    }
  }

  /** Convenience: emit a watchlist match event. */
  async watchlistMatch(
    patternId: string,
    category: string,
    severity: string,
    trigger: string,
    matchedText: string,
    context: string
  ): Promise<void> {
    const event: WatchlistEvent = {
      timestamp: new Date().toISOString(),
      type: "watchlist.match",
      patternId,
      category,
      severity,
      trigger,
      matchedText,
      context: context.slice(0, 500),
    };
    await this.emit(event);
  }

  /** Convenience: emit a gate check event. */
  async gateCheck(
    score: number,
    reason: string,
    wordCount: number,
    escalated: boolean
  ): Promise<void> {
    const event: GateEvent = {
      timestamp: new Date().toISOString(),
      type: escalated ? "gate.escalation" : "gate.check",
      score,
      reason,
      wordCount,
    };
    await this.emit(event);
  }

  /** Convenience: emit an analysis complete event. */
  async analysisComplete(
    triggerReason: string,
    insights: string
  ): Promise<void> {
    const event: AnalysisEvent = {
      timestamp: new Date().toISOString(),
      type: "analysis.complete",
      triggerReason,
      insights: insights.slice(0, 1000),
    };
    await this.emit(event);
  }

  /** Get the log file path (for display). */
  get path(): string {
    return this.logPath;
  }
}

// ── Helpers ────────────────────────────────────────────────────────

function eventIcon(type: string): string {
  switch (type) {
    case "watchlist.match":
      return "🫀";
    case "gate.check":
      return "🚦";
    case "gate.escalation":
      return "🚀";
    case "analysis.complete":
      return "📊";
    default:
      return "📡";
  }
}

function eventSummary(event: ListenEvent): string {
  switch (event.type) {
    case "watchlist.match":
      return `[${event.severity}] ${event.category}/${event.patternId}: "${event.trigger}"`;
    case "gate.check":
    case "gate.escalation":
      return `score=${event.score}/10 — ${event.reason}`;
    case "analysis.complete":
      return `(${String(event.insights).length} chars)`;
    default:
      return "";
  }
}
