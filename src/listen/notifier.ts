/**
 * Notifier — sends macOS notifications via osascript.
 *
 * Pipes AppleScript via stdin to avoid injection vulnerabilities.
 * Also logs to terminal with formatting.
 */

import type { AnalysisResult } from "./config";

/**
 * Send a macOS notification with the analysis result.
 * Uses stdin piping to osascript to prevent AppleScript injection.
 */
export async function notify(result: AnalysisResult): Promise<void> {
  const title = `🎧 listen — ${result.triggerReason}`;
  // Truncate for notification (macOS has limits)
  const body = result.insights.slice(0, 500);

  // Proper AppleScript escaping: backslashes first, then quotes
  const escapeAS = (s: string): string =>
    s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");

  const script = `display notification "${escapeAS(body)}" with title "${escapeAS(title)}" sound name "Glass"`;

  try {
    // Pipe the script via stdin to avoid shell interpretation entirely
    const proc = Bun.spawn(["osascript", "-"], {
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });
    proc.stdin.write(script);
    proc.stdin.end();
    await proc.exited;
  } catch {
    // Notification failed — non-critical, we still log to terminal
  }

  // Always print to terminal
  logInsight(result);
}

/** Pretty-print the analysis to the terminal. */
function logInsight(result: AnalysisResult): void {
  const time = result.timestamp.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
  });

  console.log();
  console.log(
    `  ┌──────────────────────────────────────────────────────────────┐`
  );
  console.log(
    `  │  🔔 INSIGHT @ ${time}  (trigger: ${result.triggerReason})`.padEnd(
      65
    ) + "│"
  );
  console.log(
    `  ├──────────────────────────────────────────────────────────────┤`
  );

  // Word-wrap the insights
  const lines = result.insights.split("\n").filter(Boolean);
  for (const line of lines) {
    const wrapped = wordWrap(line, 60);
    for (const w of wrapped) {
      console.log(`  │  ${w}`.padEnd(65) + "│");
    }
  }

  console.log(
    `  └──────────────────────────────────────────────────────────────┘`
  );
  console.log();
}

function wordWrap(text: string, maxWidth: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    if (current.length + word.length + 1 > maxWidth) {
      lines.push(current);
      current = word;
    } else {
      current = current ? `${current} ${word}` : word;
    }
  }
  if (current) lines.push(current);
  return lines.length ? lines : [""];
}
