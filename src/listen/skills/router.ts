/**
 * Skill Router — single LLM call to classify transcripts across all skills.
 *
 * Replaces the old gate.ts. Instead of "score 0-10", the router asks:
 *   "Which skills should fire? Also, how interesting is this overall?"
 *
 * Returns:
 *   - skill matches (with action, params, confidence)
 *   - interest score 0-10 (preserves the old gate escalation path)
 *   - brief reason
 *
 * The prompt is built dynamically from the SkillRegistry,
 * so adding a new skill automatically includes it in routing.
 */

import type { ListenConfig } from "../config";
import type { RouterResult, RouterContext } from "./types";
import { SkillRegistry } from "./registry";

const ROUTER_TIMEOUT_MS = 10_000;

/**
 * Build the system prompt for the router.
 * Includes all registered skill definitions.
 */
function buildSystemPrompt(registry: SkillRegistry): string {
  const skillsBlock = registry.buildSkillsPrompt();

  return `You are a skill router for a voice assistant that listens to live conversations.
Given a transcript snippet, determine:
1. Which skills (if any) should activate
2. An overall "interest" score (0-10) for the conversation content

RULES:
- Only activate a skill when there is CLEAR INTENT or a CLEAR SIGNAL.
- "play some music" → music.play. "I was talking about music" → NO activation.
- IMPORTANT: Multiple skills CAN and SHOULD fire from one transcript when multiple intents are present.
  Example: "skip this song, I'm so stupid" → BOTH music.skip AND wellbeing.check_in.
- Include ALL matching skills in the array, not just the primary one.
- If nothing matches, return an empty skills array.
- Interest score: ACTION ITEMS (7+), DECISIONS (7+), KEY INSIGHTS (6+), OPEN QUESTIONS (5+), RED FLAGS (8+), small talk (0-2).

CRITICAL — wellbeing skill false positive rules:
- ONLY activate wellbeing when the SPEAKER is talking about THEMSELVES.
- "I'm so stupid" → wellbeing YES (first person, about self).
- "he said he was stupid" → wellbeing NO (third person, about someone else).
- "she's exhausted and wants to quit" → wellbeing NO (talking about another person).
- "I'm a bit tired, had a late night" → wellbeing NO (casual/mundane, not distress).
- The speaker must express GENUINE DISTRESS about themselves, not report someone else's state or mention tiredness casually.

Available skills:
${skillsBlock}

Respond with ONLY a JSON object, nothing else:
{"skills": [{"skill": "name", "action": "action_name", "params": {}, "confidence": 0.9}], "interest": 2, "reason": "brief explanation"}`;
}

/**
 * Route a transcript through the skill router.
 * Makes a single LLM call that handles both skill classification and interest scoring.
 */
export async function routeTranscript(
  ctx: RouterContext,
  registry: SkillRegistry,
  config: ListenConfig
): Promise<RouterResult> {
  // Skip trivially short transcripts
  if (!ctx.transcript || ctx.transcript.trim().length < 10) {
    return { matches: [], interest: 0, reason: "too short" };
  }

  const systemPrompt = buildSystemPrompt(registry);

  // Build user message with all available context
  const parts: string[] = [];

  // Recent conversation
  if (ctx.buffer) {
    parts.push(`RECENT CONTEXT:\n---\n${ctx.buffer}\n---`);
  }

  // Recent skill executions (so the router knows what just happened)
  if (ctx.recentSkills?.length) {
    const historyLines = ctx.recentSkills.map((s) => {
      const ago = Math.round((Date.now() - s.timestamp.getTime()) / 1000);
      return `  ${s.skill}.${s.action} (${ago}s ago, ${s.success ? "succeeded" : "failed"})${s.voice ? ` → "${s.voice}"` : ""}`;
    });
    parts.push(`RECENTLY EXECUTED SKILLS:\n${historyLines.join("\n")}`);
  }

  // Current transcript
  parts.push(`LATEST TRANSCRIPT:\n---\n${ctx.transcript}\n---`);

  const userMessage = parts.join("\n\n");

  try {
    const raw = config.localGateEndpoint
      ? await routeLocal(systemPrompt, userMessage, config)
      : await routeRemote(systemPrompt, userMessage, config);

    return parseRouterResponse(raw);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("  ⚠ router failed:", msg);
    return { matches: [], interest: 0, reason: "router error" };
  }
}

// ── Helpers ───────────────────────────────────────────────────────

/** Extract the first balanced JSON object from a string. */
function extractFirstJson(raw: string): string | null {
  const start = raw.indexOf("{");
  if (start === -1) return null;
  let depth = 0;
  for (let i = start; i < raw.length; i++) {
    if (raw[i] === "{") depth++;
    else if (raw[i] === "}") depth--;
    if (depth === 0) return raw.slice(start, i + 1);
  }
  return null;
}

/** Coerce all param values to strings (LLM may return numbers, etc.). */
function sanitizeParams(params: unknown): Record<string, string> {
  if (!params || typeof params !== "object") return {};
  return Object.fromEntries(
    Object.entries(params as Record<string, unknown>).map(([k, v]) => [
      k,
      String(v ?? ""),
    ])
  );
}

// ── Parse response ────────────────────────────────────────────────

function parseRouterResponse(raw: string): RouterResult {
  // Extract first balanced JSON object from response
  const jsonStr = extractFirstJson(raw);
  if (!jsonStr) {
    return { matches: [], interest: 0, reason: "parse error: no JSON found" };
  }

  try {
    const parsed = JSON.parse(jsonStr);

    const matches = Array.isArray(parsed.skills)
      ? parsed.skills
          .filter(
            (s: Record<string, unknown>) =>
              typeof s.skill === "string" &&
              typeof s.action === "string" &&
              typeof s.confidence === "number" &&
              s.confidence >= 0.5 // filter low-confidence noise
          )
          .map((s: Record<string, unknown>) => ({
            skill: String(s.skill),
            action: String(s.action),
            params: sanitizeParams(s.params),
            confidence: Number(s.confidence),
          }))
      : [];

    return {
      matches,
      interest: Math.max(0, Math.min(10, Number(parsed.interest) || 0)),
      reason: String(parsed.reason || "unknown"),
    };
  } catch {
    return { matches: [], interest: 0, reason: "parse error: invalid JSON" };
  }
}

// ── Local backend (LM Studio / ollama) ────────────────────────────

async function routeLocal(
  systemPrompt: string,
  userMessage: string,
  config: ListenConfig
): Promise<string> {
  const endpoint = `${config.localGateEndpoint}/v1/chat/completions`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ROUTER_TIMEOUT_MS);

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: config.gateModel,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userMessage },
        ],
        max_tokens: 200,
        temperature: 0,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      throw new Error(`LM Studio ${res.status}: ${await res.text()}`);
    }

    const data = (await res.json()) as {
      choices: { message: { content: string } }[];
    };
    return data.choices[0]?.message?.content ?? "";
  } finally {
    clearTimeout(timer);
  }
}

// ── Remote backend (opencode) ─────────────────────────────────────

async function routeRemote(
  systemPrompt: string,
  userMessage: string,
  config: ListenConfig
): Promise<string> {
  const prompt = `${systemPrompt}\n\n${userMessage}`;
  const tmpFile = `/tmp/listen-router-${Date.now()}.txt`;
  await Bun.write(tmpFile, prompt);

  const proc = Bun.spawn(
    ["opencode", "run", "-m", config.gateModel, "-f", tmpFile, "--", "Route this transcript."],
    { stdout: "pipe", stderr: "pipe" }
  );

  const timer = setTimeout(() => proc.kill(), ROUTER_TIMEOUT_MS);

  try {
    // Read both streams concurrently to avoid pipe buffer deadlock
    const [output, stderrText] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      throw new Error(`opencode exited ${exitCode}: ${stderrText.slice(0, 200)}`);
    }
    return output;
  } finally {
    clearTimeout(timer);
    try { await Bun.file(tmpFile).unlink(); } catch {}
  }
}
