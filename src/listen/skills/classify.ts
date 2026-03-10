/**
 * Skill Classifier — broadcasts transcripts to per-skill MLX experts in parallel.
 *
 * Architecture:
 *   Each skill has its own fine-tuned Qwen3.5-0.8B binary classifier served
 *   by the expert server (Python/MLX) on per-skill endpoints.
 *
 *   This module fires one HTTP request per skill via Promise.all, so all
 *   experts classify concurrently. The threaded Python server handles each
 *   request in its own thread.
 *
 * Observability:
 *   Every classification returns per-expert metrics (latency, status,
 *   confidence) plus aggregate timing so you can see the parallelism gain.
 *
 * Returns:
 *   - skill matches (with action, params, confidence)
 *   - per-expert classification details (ExpertClassification[])
 *   - interest score 0-10
 *   - wall-clock time + sum-of-expert time (shows parallel speedup)
 *   - brief reason
 */

import type { ListenConfig } from "../config";
import type { RouterResult, RouterContext, SkillMatch } from "./types";
import type { SkillRegistry } from "./registry";

const EXPERT_TIMEOUT_MS = 8_000; // per-expert timeout

// ── Per-expert classification result ──────────────────────────────

/** Full observability record for a single expert's classification. */
export interface ExpertClassification {
  /** Skill name (e.g. "accommodator", "wellbeing") */
  skill: string;
  /** Did this expert detect a match? */
  match: boolean;
  /** Action name if matched */
  action?: string;
  /** Confidence 0.0-1.0 if matched */
  confidence?: number;
  /** Expert-side inference latency (ms) — from the Python server */
  expertMs: number;
  /** Round-trip latency including HTTP overhead (ms) — from Bun */
  roundTripMs: number;
  /** Classification status */
  status: "ok" | "error" | "timeout" | "too_short";
  /** Error message if status is "error" */
  error?: string;
}

/** Extended result with per-expert observability. */
export interface ClassifyResult extends RouterResult {
  /** Per-expert classification details */
  experts: ExpertClassification[];
  /** Wall-clock time for the full parallel fan-out (ms) */
  classifyMs: number;
  /** Sum of individual expert round-trip times (ms) — shows parallelism gain */
  expertSumMs: number;
}

// ── Main classify function ────────────────────────────────────────

/**
 * Classify a transcript by broadcasting to all skill experts in parallel.
 *
 * Each registered skill gets its own HTTP request to the expert server.
 * Results are gathered via Promise.all — no expert blocks another.
 */
export async function classifyTranscript(
  ctx: RouterContext,
  registry: SkillRegistry,
  config: ListenConfig
): Promise<ClassifyResult> {
  // Skip trivially short transcripts
  if (!ctx.transcript || ctx.transcript.trim().length < 10) {
    return {
      matches: [],
      interest: 0,
      reason: "too short",
      experts: [],
      classifyMs: 0,
      expertSumMs: 0,
    };
  }

  try {
    return await fanOutClassify(ctx, registry, config);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("  ⚠ classify failed:", msg);
    return {
      matches: [],
      interest: 0,
      reason: `classify error: ${msg}`,
      experts: [],
      classifyMs: 0,
      expertSumMs: 0,
    };
  }
}

// ── Parallel classify — workers run concurrently ──────────────────
//
// The expert server runs one worker process per skill. Each worker
// has its own Metal context (separate GPU command buffers). When we
// call POST /v1/classify, all workers classify the transcript
// simultaneously → real ~2× speedup on Apple Silicon.
//
// The server returns per-expert timing + wall/sum/gain metrics.

async function fanOutClassify(
  ctx: RouterContext,
  _registry: SkillRegistry,
  config: ListenConfig
): Promise<ClassifyResult> {
  const endpoint = `${config.expertEndpoint}/v1/classify`;
  const wallStart = performance.now();

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), EXPERT_TIMEOUT_MS);

  try {
    const body: Record<string, unknown> = {
      transcript: ctx.transcript,
    };

    if (ctx.buffer) {
      body.context = ctx.buffer;
    }

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const roundTripMs = Math.round(performance.now() - wallStart);

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`expert server ${res.status}: ${text.slice(0, 200)}`);
    }

    const data = (await res.json()) as {
      results?: ExpertServerResult[];
      reason?: string;
      wall_ms?: number;
      expert_sum_ms?: number;
      parallel_gain?: number;
    };
    clearTimeout(timer);

    // Parse per-expert results into our observability format
    const classifications: ExpertClassification[] = (data.results || []).map(
      (r) => ({
        skill: r.skill,
        match: Boolean(r.match),
        action: r.match ? r.action || "" : undefined,
        confidence: r.match ? r.confidence || 0 : undefined,
        expertMs: r.latency_ms || 0,
        roundTripMs,
        status: (r.status as ExpertClassification["status"]) || "ok",
        error: r.error,
      })
    );

    // Use server-side parallel timing (workers run concurrently)
    const classifyMs = roundTripMs;
    const expertSumMs = data.expert_sum_ms
      ? Math.round(data.expert_sum_ms)
      : Math.round(classifications.reduce((sum, c) => sum + c.expertMs, 0));

    // Build matches from experts that returned a positive classification
    const matches: SkillMatch[] = classifications
      .filter((c) => c.match && c.status === "ok")
      .map((c) => ({
        skill: c.skill,
        action: c.action || "",
        params: {} as Record<string, string>,
        confidence: c.confidence || 0.9,
      }));

    const interest = computeInterest(matches);

    const reason =
      matches.length > 0
        ? matches.map((m) => `${m.skill}.${m.action}`).join(" + ")
        : "no skills matched";

    return {
      matches,
      interest,
      reason,
      experts: classifications,
      classifyMs,
      expertSumMs,
    };
  } catch (err) {
    clearTimeout(timer);
    const isAbort = err instanceof Error && err.name === "AbortError";
    throw new Error(
      isAbort
        ? `timeout after ${EXPERT_TIMEOUT_MS}ms`
        : err instanceof Error
          ? err.message
          : String(err)
    );
  }
}

/** Raw result from the expert server's /v1/classify endpoint. */
interface ExpertServerResult {
  skill: string;
  match: boolean;
  action?: string;
  confidence?: number;
  latency_ms?: number;
  status?: string;
  error?: string;
}

// ── Interest scoring ──────────────────────────────────────────────

function computeInterest(matches: SkillMatch[]): number {
  let interest = 1;
  for (const match of matches) {
    if (match.skill === "wellbeing") {
      interest = Math.max(interest, 8);
    } else if (match.skill === "accommodator") {
      interest = Math.max(interest, 5);
    } else {
      interest = Math.max(interest, 4);
    }
  }
  // Multiple skills firing = higher interest
  if (matches.length > 1) {
    interest = Math.min(10, interest + 1);
  }
  return interest;
}
