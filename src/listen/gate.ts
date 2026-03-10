/**
 * Gate — decides if transcript is worth escalating.
 *
 * Two backends:
 *   - "local"  → direct fetch() to LM Studio / ollama OpenAI-compatible API (~0.5s)
 *   - "remote" → opencode CLI subprocess to cloud model (~5-8s)
 *
 * Auto-detects: if gateModel starts with "http" or config has a local endpoint,
 * uses local. Otherwise falls back to opencode.
 */

import type { ListenConfig, GateResult } from "./config";

const GATE_TIMEOUT_MS = 10_000;

const GATE_SYSTEM = `You are a conversation gate. Score transcript segments 0-10.
Criteria: ACTION ITEMS (7+), DECISIONS (7+), KEY INSIGHTS (6+), OPEN QUESTIONS (5+), RED FLAGS (8+), TECHNICAL DETAILS (6+), small talk/silence (0-2).
Respond with ONLY a JSON object, nothing else: {"score": N, "reason": "10 words max"}`;

/**
 * Run the gate check on a transcript segment.
 * Auto-selects local (LM Studio) or remote (opencode) backend.
 */
export async function runGate(
  text: string,
  config: ListenConfig
): Promise<GateResult> {
  if (!text || text.length < 20) {
    return { score: 0, reason: "too short / silence" };
  }

  try {
    const raw = config.localGateEndpoint
      ? await gateLocal(text, config)
      : await gateOpencode(text, config);

    return parseGateResponse(raw);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("  ⚠ gate check failed:", message);
    return { score: 0, reason: "gate error" };
  }
}

// ── Parse response ─────────────────────────────────────────────────

function parseGateResponse(raw: string): GateResult {
  // Try to extract JSON
  const jsonMatch = raw.match(/\{[^}]+\}/);
  if (jsonMatch) {
    const parsed = JSON.parse(jsonMatch[0]);
    return {
      score: Math.max(0, Math.min(10, Number(parsed.score) || 0)),
      reason: String(parsed.reason || "unknown"),
    };
  }

  // Fallback: bare number
  const numMatch = raw.match(/\b(\d{1,2})\b/);
  return {
    score: numMatch ? Math.min(10, Number(numMatch[1])) : 0,
    reason: "parse-fallback",
  };
}

// ── Local backend (LM Studio / ollama) ─────────────────────────────

async function gateLocal(
  text: string,
  config: ListenConfig
): Promise<string> {
  const endpoint = `${config.localGateEndpoint}/v1/chat/completions`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GATE_TIMEOUT_MS);

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: config.gateModel,
        messages: [
          { role: "system", content: GATE_SYSTEM },
          { role: "user", content: `TRANSCRIPT:\n---\n${text}\n---` },
        ],
        max_tokens: 50,
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

// ── Remote backend (opencode CLI) ──────────────────────────────────

const GATE_MESSAGE_OPENCODE = `You are a conversation monitor. The attached file contains a conversation transcript. Score it 0-10: ACTION ITEMS (7+), DECISIONS (7+), KEY INSIGHTS (6+), OPEN QUESTIONS (5+), RED FLAGS (8+), TECHNICAL DETAILS (6+), small talk (0-2). Respond with ONLY a JSON object: {"score": N, "reason": "10 words max"}`;

async function gateOpencode(
  text: string,
  config: ListenConfig
): Promise<string> {
  const tmpFile = `/tmp/listen-prompt-${Date.now()}.txt`;
  await Bun.write(tmpFile, text);

  const proc = Bun.spawn(
    [
      "opencode",
      "run",
      "-m",
      config.gateModel,
      "-f",
      tmpFile,
      "--",
      GATE_MESSAGE_OPENCODE,
    ],
    { stdout: "pipe", stderr: "pipe" }
  );

  const timer = setTimeout(() => proc.kill(), GATE_TIMEOUT_MS);

  try {
    const output = await new Response(proc.stdout).text();
    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      const stderr = await new Response(proc.stderr).text();
      throw new Error(`opencode exited ${exitCode}: ${stderr.slice(0, 200)}`);
    }

    return output;
  } finally {
    clearTimeout(timer);
    try {
      await Bun.file(tmpFile).exists() &&
        Bun.spawn(["rm", tmpFile], { stdout: "ignore", stderr: "ignore" });
    } catch {}
  }
}
