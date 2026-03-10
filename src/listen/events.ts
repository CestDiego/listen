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

export interface RouterEvent extends ListenEvent {
  type: "router.result";
  interest: number;
  reason: string;
  matchedSkills: string[];
  wordCount: number;
}

export interface ClassifyEvent extends ListenEvent {
  type: "classify.fanout";
  /** Wall-clock time for the full parallel fan-out (ms) */
  classifyMs: number;
  /** Sum of individual expert round-trip times (ms) */
  expertSumMs: number;
  /** Parallelism gain ratio (expertSumMs / classifyMs) */
  parallelGain: string;
  /** Per-expert breakdown */
  experts: Array<{
    skill: string;
    match: boolean;
    action?: string;
    confidence?: number;
    expertMs: number;
    roundTripMs: number;
    status: string;
    error?: string;
  }>;
  /** Skills that matched */
  matchedSkills: string[];
  /** Interest score */
  interest: number;
}

export interface SkillEvent extends ListenEvent {
  type: "skill.executed";
  skill: string;
  action: string;
  success: boolean;
  confidence: number;
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

  /** Convenience: emit a router result event (legacy, still used for timeline). */
  async routerResult(
    interest: number,
    reason: string,
    matchedSkills: string[],
    wordCount: number
  ): Promise<void> {
    const event: RouterEvent = {
      timestamp: new Date().toISOString(),
      type: "router.result",
      interest,
      reason,
      matchedSkills,
      wordCount,
    };
    await this.emit(event);
  }

  /** Convenience: emit a classify fan-out event with full per-expert observability. */
  async classifyFanout(data: {
    classifyMs: number;
    expertSumMs: number;
    experts: ClassifyEvent["experts"];
    matchedSkills: string[];
    interest: number;
  }): Promise<void> {
    const gain = data.classifyMs > 0
      ? (data.expertSumMs / data.classifyMs).toFixed(2)
      : "1.00";
    const event: ClassifyEvent = {
      timestamp: new Date().toISOString(),
      type: "classify.fanout",
      classifyMs: data.classifyMs,
      expertSumMs: data.expertSumMs,
      parallelGain: `${gain}×`,
      experts: data.experts,
      matchedSkills: data.matchedSkills,
      interest: data.interest,
    };
    await this.emit(event);
  }

  /** Convenience: emit a skill execution event. */
  async skillExecuted(
    skill: string,
    action: string,
    success: boolean,
    confidence: number
  ): Promise<void> {
    const event: SkillEvent = {
      timestamp: new Date().toISOString(),
      type: "skill.executed",
      skill,
      action,
      success,
      confidence,
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
    case "router.result":
      return "🧭";
    case "classify.fanout":
      return "🎯";
    case "skill.executed":
      return "⚡";
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
    case "router.result":
      return `interest=${event.interest}/10 skills=[${event.matchedSkills}] — ${event.reason}`;
    case "classify.fanout": {
      const e = event as unknown as ClassifyEvent;
      const experts = e.experts.map((x) => {
        const icon = x.match ? "+" : "-";
        const action = x.match ? `.${x.action}` : "";
        const conf = x.confidence ? ` [${(x.confidence * 100).toFixed(0)}%]` : "";
        return `${icon}${x.skill}${action}${conf} ${x.roundTripMs}ms`;
      }).join(", ");
      return `${e.classifyMs}ms wall (${e.parallelGain} gain) [${experts}]`;
    }
    case "skill.executed":
      return `${event.skill}.${event.action} ${event.success ? "✓" : "✗"}`;
    case "analysis.complete":
      return `(${String(event.insights).length} chars)`;
    default:
      return "";
  }
}
