/**
 * inject-test.ts — POST a timed sequence of transcripts to the Listen dashboard
 * to exercise the intent-vector visualization (radar chart / Intent tab).
 *
 * Usage:
 *   bun run scripts/inject-test.ts [--url http://localhost:3838]
 *
 * Prerequisites:
 *   - Dashboard running (./start.sh)
 *   - Expert server running at localhost:8234
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DEFAULT_URL = "http://localhost:3838";

function parseArgs(): { url: string } {
  const args = Bun.argv.slice(2);
  let url = DEFAULT_URL;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--url" && args[i + 1]) {
      url = args[i + 1];
      i++;
    }
  }
  // Strip trailing slash
  return { url: url.replace(/\/+$/, "") };
}

// ---------------------------------------------------------------------------
// Test sequence
// ---------------------------------------------------------------------------

interface Step {
  text: string | null; // null = wait-only step
  delayBeforeMs: number;
  expectedEffect: string;
}

const SEQUENCE: Step[] = [
  {
    text: "play some jazz music",
    delayBeforeMs: 0,
    expectedEffect: "Music activates",
  },
  {
    text: "turn the volume up",
    delayBeforeMs: 2000,
    expectedEffect: "Music stays high",
  },
  {
    text: "I'm so tired of everything",
    delayBeforeMs: 3000,
    expectedEffect: "Wellbeing activates. Music starts decaying",
  },
  {
    text: "I feel like I can't do anything right",
    delayBeforeMs: 2000,
    expectedEffect: "Wellbeing stays high",
  },
  {
    text: "skip this song",
    delayBeforeMs: 4000,
    expectedEffect: "Music re-activates. Wellbeing decaying but still present",
  },
  {
    text: "what's the weather like today",
    delayBeforeMs: 3000,
    expectedEffect: "Neutral. Both start decaying",
  },
  {
    text: null,
    delayBeforeMs: 8000,
    expectedEffect: "Watch decay on radar chart",
  },
  {
    text: "I don't think I'm good enough",
    delayBeforeMs: 0,
    expectedEffect: "Wellbeing spikes back up",
  },
  {
    text: "play the next track",
    delayBeforeMs: 2000,
    expectedEffect: "Music + wellbeing both active (dual)",
  },
  {
    text: "that sounds great, thanks",
    delayBeforeMs: 5000,
    expectedEffect: "Neutral. Final decay",
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function timestamp(): string {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const { url } = parseArgs();

  console.log("┌─────────────────────────────────────────────────────────┐");
  console.log("│  inject-test — Intent Vector Visualization Test         │");
  console.log("│                                                         │");
  console.log("│  Posts a timed sequence of transcripts to the Listen    │");
  console.log("│  dashboard to exercise the classify pipeline and        │");
  console.log("│  intent radar chart.                                    │");
  console.log("│                                                         │");
  console.log(`│  Target: ${url.padEnd(47)}│`);
  console.log("│  Endpoint: POST /api/transcript                        │");
  console.log("│  Mode: moonshine (full classify pipeline)              │");
  console.log("└─────────────────────────────────────────────────────────┘");
  console.log();

  // -- Preflight check: is the dashboard reachable? -------------------------
  try {
    const probe = await fetch(`${url}/api/session`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!probe.ok) {
      console.error(
        `Dashboard returned ${probe.status} on /api/session — is it running?`
      );
      console.error(`  Start it with:  ./start.sh`);
      process.exit(1);
    }
  } catch {
    console.error(`Cannot reach dashboard at ${url}`);
    console.error(`  Make sure the dashboard is running:  ./start.sh`);
    console.error(
      `  And the expert server is running on localhost:8234`
    );
    process.exit(1);
  }

  console.log("Dashboard is reachable. Starting sequence...\n");

  // -- Open dashboard in browser --------------------------------------------
  Bun.spawn(["open", url]);

  // -- Run the sequence -----------------------------------------------------
  let lineId = 1;

  for (const step of SEQUENCE) {
    // Delay
    if (step.delayBeforeMs > 0) {
      console.log(
        `[${timestamp()}]    … waiting ${step.delayBeforeMs}ms (${step.expectedEffect})`
      );
      await sleep(step.delayBeforeMs);
    }

    // Wait-only step (no POST)
    if (step.text === null) {
      continue;
    }

    // POST transcript
    const body = {
      text: step.text,
      durationSeconds: 5,
      lineId,
      source: "inject-test",
    };

    console.log(
      `[${timestamp()}] -> "${step.text}" (expected: ${step.expectedEffect})`
    );

    try {
      const res = await fetch(`${url}/api/transcript`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(5000),
      });
      console.log(`[${timestamp()}]    status: ${res.status}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(`[${timestamp()}]    WARNING: POST failed — ${msg}`);
    }

    lineId++;
    console.log();
  }

  // -- Done -----------------------------------------------------------------
  console.log("Done. Watch the Intent tab in the dashboard.");
}

main();
