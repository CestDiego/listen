/**
 * Skill Router — routes transcripts through per-skill MLX expert models.
 *
 * Architecture:
 *   Each skill has its own fine-tuned Qwen3.5-0.8B binary classifier.
 *   The expert server (Python/MLX) runs all experts and returns results.
 *   This replaces the old single-LLM-call approach via LM Studio.
 *
 * Returns:
 *   - skill matches (with action, params, confidence)
 *   - interest score 0-10
 *   - brief reason
 *
 * The experts run in parallel on Apple Silicon via MLX — ~270ms total.
 */

import type { ListenConfig } from "../config";
import type { RouterResult, RouterContext } from "./types";

const ROUTER_TIMEOUT_MS = 10_000;

// ── Expert server types ───────────────────────────────────────────

interface ExpertResult {
  skill: string;
  match: boolean;
  action?: string;
  confidence?: number;
  latency_ms?: number;
  error?: string;
}

interface ExpertResponse {
  results: ExpertResult[];
  reason?: string;
}

/**
 * Route a transcript through the per-skill expert models.
 *
 * Calls the MLX expert server which runs each skill's fine-tuned
 * classifier in parallel. No LM Studio needed.
 */
export async function routeTranscript(
  ctx: RouterContext,
  _registry: unknown, // kept for API compat, experts are self-contained
  config: ListenConfig
): Promise<RouterResult> {
  // Skip trivially short transcripts
  if (!ctx.transcript || ctx.transcript.trim().length < 10) {
    return { matches: [], interest: 0, reason: "too short" };
  }

  try {
    return await routeViaExperts(ctx, config);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("  ⚠ router failed:", msg);
    return { matches: [], interest: 0, reason: "router error" };
  }
}

// ── Expert server backend ─────────────────────────────────────────

async function routeViaExperts(
  ctx: RouterContext,
  config: ListenConfig
): Promise<RouterResult> {
  const endpoint = `${config.expertEndpoint}/v1/classify`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ROUTER_TIMEOUT_MS);

  try {
    const body: Record<string, unknown> = {
      transcript: ctx.transcript,
    };

    // Pass context to the expert server for richer classification
    if (ctx.buffer) {
      body.context = ctx.buffer;
    }

    // Pass recent skill executions so experts can avoid re-triggers
    if (ctx.recentSkills?.length) {
      body.recent_skills = ctx.recentSkills.map((s) => ({
        skill: s.skill,
        action: s.action,
        ago_seconds: Math.round((Date.now() - s.timestamp.getTime()) / 1000),
        voice: s.voice,
      }));
    }

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!res.ok) {
      throw new Error(`expert server ${res.status}: ${await res.text()}`);
    }

    const data = (await res.json()) as ExpertResponse;
    return parseExpertResponse(data);
  } finally {
    clearTimeout(timer);
  }
}

// ── Parse expert response into RouterResult ───────────────────────

function parseExpertResponse(data: ExpertResponse): RouterResult {
  const matches = (data.results || [])
    .filter((r) => r.match && !r.error)
    .map((r) => ({
      skill: r.skill,
      action: r.action || "",
      params: {} as Record<string, string>,
      confidence: r.confidence || 0.9,
    }));

  // Compute interest score from expert results:
  //   - Each matching skill adds to interest
  //   - Wellbeing matches always high interest (8+)
  //   - Accommodator matches moderate interest (5)
  //   - No matches = low interest (1-2)
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

  // Build reason string
  const reason =
    matches.length > 0
      ? matches.map((m) => `${m.skill}.${m.action}`).join(" + ")
      : "no skills matched";

  return { matches, interest, reason };
}
