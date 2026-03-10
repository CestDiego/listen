/**
 * Analyzer — calls a big model via opencode to extract insights
 * from the accumulated transcript.
 *
 * Uses Bun.spawn + temp file to avoid shell injection and ARG_MAX limits.
 */

import type { ListenConfig, AnalysisResult } from "./config";

const ANALYSIS_TIMEOUT_MS = 60_000;

// Instructions go as CLI message (safe, fixed-length).
// Transcript goes in the attached file (avoids ARG_MAX / injection).
const ANALYSIS_MESSAGE = `You are an expert meeting analyst. The attached file is a conversation transcript. Extract actionable intelligence. Be EXTREMELY concise (under 280 chars total, like a tweet). Use this format — omit empty categories: 💡 INSIGHTS: <1-2 non-obvious bullet points> ✅ ACTIONS: <tasks, commitments> ⚠️ FLAGS: <risks, contradictions, open questions>. Output ONLY the formatted result, no preamble.`;

/**
 * Run full analysis on the transcript buffer.
 */
export async function analyze(
  transcript: string,
  triggerReason: string,
  config: ListenConfig
): Promise<AnalysisResult> {
  try {
    const result = await runOpencode(
      transcript,
      ANALYSIS_MESSAGE,
      config.analysisModel,
      ANALYSIS_TIMEOUT_MS
    );

    // Strip any ANSI escape codes from opencode output
    const clean = result
      .replace(
        /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g,
        ""
      )
      .trim();

    return {
      insights: clean || "No actionable insights found.",
      timestamp: new Date(),
      triggerReason,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return {
      insights: `Analysis failed: ${message}`,
      timestamp: new Date(),
      triggerReason,
    };
  }
}

/**
 * Run opencode via Bun.spawn, piping prompt through a temp file.
 * Avoids shell injection and ARG_MAX limits.
 */
async function runOpencode(
  transcript: string,
  message: string,
  model: string,
  timeoutMs: number
): Promise<string> {
  // Transcript goes in file (untrusted, can be large)
  // Instructions go as CLI message (safe, fixed-length)
  const tmpFile = `/tmp/listen-analysis-${Date.now()}.txt`;
  await Bun.write(tmpFile, transcript);

  const proc = Bun.spawn(
    ["opencode", "run", "-m", model, "-f", tmpFile, "--", message],
    {
      stdout: "pipe",
      stderr: "pipe",
    }
  );

  const timer = setTimeout(() => proc.kill(), timeoutMs);

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
