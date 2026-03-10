/**
 * inject-test.ts — POST a timed sequence of transcripts to the Listen dashboard
 * to exercise the intent-vector visualization (radar chart / Intent tab).
 *
 * Usage:
 *   bun run scripts/inject-test.ts [--url http://localhost:3838] [--scenario basic|gate|all]
 *
 * Scenarios:
 *   basic  — Original mixed-intent sequence (music + wellbeing interleaved)
 *   gate   — ActivationGate hysteresis test (ambiguous before/after trigger)
 *   all    — Runs basic, then a 10s gap, then gate (default)
 *
 * Prerequisites:
 *   - Dashboard running (./start.sh)
 *   - Expert server running at localhost:8234
 */

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DEFAULT_URL = "http://localhost:3838";

type Scenario = "basic" | "gate" | "all";

function parseArgs(): { url: string; scenario: Scenario } {
  const args = Bun.argv.slice(2);
  let url = DEFAULT_URL;
  let scenario: Scenario = "all";
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--url" && args[i + 1]) {
      url = args[i + 1];
      i++;
    }
    if (args[i] === "--scenario" && args[i + 1]) {
      const val = args[i + 1];
      if (val === "basic" || val === "gate" || val === "all") {
        scenario = val;
      } else {
        console.error(
          `Unknown scenario "${val}" — expected basic, gate, or all`
        );
        process.exit(1);
      }
      i++;
    }
  }
  // Strip trailing slash
  return { url: url.replace(/\/+$/, ""), scenario };
}

// ---------------------------------------------------------------------------
// Test sequences
// ---------------------------------------------------------------------------

interface Step {
  text: string | null; // null = wait-only step
  delayBeforeMs: number;
  expectedEffect: string;
  gateState?: string; // optional gate annotation for gate sequence
}

const BASIC_SEQUENCE: Step[] = [
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

const GATE_SEQUENCE: Step[] = [
  // Phase A: Establish baseline — ambiguous input WITHOUT prior wellbeing
  {
    text: "I'm just tired",
    delayBeforeMs: 0,
    expectedEffect:
      "BASELINE: Should NOT trigger wellbeing (ambiguous, no prior context)",
    gateState: "[gate: idle]",
  },
  {
    text: "whatever, it doesn't matter",
    delayBeforeMs: 2000,
    expectedEffect:
      "BASELINE: Should NOT trigger wellbeing (ambiguous, no prior context)",
    gateState: "[gate: idle]",
  },

  // Phase B: Trigger wellbeing genuinely
  {
    text: "I feel like such a failure, I can't do anything right",
    delayBeforeMs: 3000,
    expectedEffect: "TRIGGER: Should activate wellbeing (clear distress signal)",
    gateState: "[gate: → active]",
  },

  // Phase C: Ambiguous inputs AFTER trigger — gate should promote these
  {
    text: "I'm just tired",
    delayBeforeMs: 2000,
    expectedEffect:
      "GATE TEST: Same text as baseline, but NOW should trigger wellbeing (gate is vigilant)",
    gateState: "[gate: vigilant, promoting]",
  },
  {
    text: "whatever, it doesn't matter",
    delayBeforeMs: 2000,
    expectedEffect:
      "GATE TEST: Same text as baseline, but NOW should trigger wellbeing (gate is vigilant)",
    gateState: "[gate: vigilant, promoting]",
  },
  {
    text: "I don't know anymore",
    delayBeforeMs: 2000,
    expectedEffect: "GATE TEST: Ambiguous, gate should promote to wellbeing",
    gateState: "[gate: vigilant, promoting]",
  },

  // Phase D: Neutral inputs — gate should eventually deactivate
  {
    text: "play some music",
    delayBeforeMs: 3000,
    expectedEffect:
      "NEUTRAL: Music command, gate still vigilant but music takes priority",
    gateState: "[gate: vigilant → idle]",
  },
  {
    text: "what time is it",
    delayBeforeMs: 3000,
    expectedEffect: "NEUTRAL: No skill match, gate still decaying",
    gateState: "[gate: vigilant → idle]",
  },
  {
    text: null,
    delayBeforeMs: 15000,
    expectedEffect:
      "DECAY: 15s pause — wellbeing vector should decay significantly",
    gateState: "[gate: vigilant → idle]",
  },
  {
    text: "I'm just tired",
    delayBeforeMs: 0,
    expectedEffect:
      "POST-DECAY: Same ambiguous text — gate should be back to idle, should NOT trigger wellbeing",
    gateState: "[gate: idle]",
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
// Sequence runner
// ---------------------------------------------------------------------------

async function runSequence(
  url: string,
  steps: Step[],
  startLineId: number
): Promise<number> {
  let lineId = startLineId;

  for (const step of steps) {
    // Delay
    if (step.delayBeforeMs > 0) {
      const gateTag = step.gateState ? ` ${step.gateState}` : "";
      console.log(
        `[${timestamp()}]    … waiting ${step.delayBeforeMs}ms (${step.expectedEffect})${gateTag}`
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

    const gateTag = step.gateState ? ` ${step.gateState}` : "";
    console.log(
      `[${timestamp()}] -> "${step.text}"${gateTag}`
    );
    console.log(
      `[${timestamp()}]    expected: ${step.expectedEffect}`
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

  return lineId;
}

// ---------------------------------------------------------------------------
// Headers
// ---------------------------------------------------------------------------

function printBasicHeader() {
  console.log(
    "───────────────────────────────────────────────────"
  );
  console.log(
    "  BASIC TEST — Mixed Intent Sequence"
  );
  console.log(
    "───────────────────────────────────────────────────"
  );
  console.log();
}

function printGateHeader() {
  console.log(
    "═══════════════════════════════════════════════════"
  );
  console.log(
    "  GATE TEST — Activation Gate Hysteresis Scenario"
  );
  console.log(
    ""
  );
  console.log(
    '  Tests that ambiguous inputs ("I\'m just tired")'
  );
  console.log(
    "  route to wellbeing ONLY after a genuine trigger,"
  );
  console.log(
    "  and stop routing after decay."
  );
  console.log(
    "═══════════════════════════════════════════════════"
  );
  console.log();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const { url, scenario } = parseArgs();

  console.log("┌─────────────────────────────────────────────────────────┐");
  console.log("│  inject-test — Intent Vector Visualization Test         │");
  console.log("│                                                         │");
  console.log("│  Posts a timed sequence of transcripts to the Listen    │");
  console.log("│  dashboard to exercise the classify pipeline and        │");
  console.log("│  intent radar chart.                                    │");
  console.log("│                                                         │");
  console.log(`│  Target:   ${url.padEnd(45)}│`);
  console.log(`│  Scenario: ${scenario.padEnd(45)}│`);
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

  // -- Run the selected scenario(s) -----------------------------------------
  let lineId = 1;

  if (scenario === "basic" || scenario === "all") {
    printBasicHeader();
    lineId = await runSequence(url, BASIC_SEQUENCE, lineId);
  }

  if (scenario === "all") {
    console.log(`[${timestamp()}]    … 10s gap between scenarios`);
    await sleep(10000);
    console.log();
  }

  if (scenario === "gate" || scenario === "all") {
    printGateHeader();
    lineId = await runSequence(url, GATE_SEQUENCE, lineId);
  }

  // -- Done -----------------------------------------------------------------
  console.log("Done. Watch the Intent tab in the dashboard.");
  if (scenario === "gate" || scenario === "all") {
    console.log(
      "Gate test: compare baseline (Phase A) vs post-trigger (Phase C) vs post-decay (Phase D)."
    );
  }
}

main();
