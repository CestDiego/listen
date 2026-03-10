#!/usr/bin/env bun
/**
 * Skill Router Eval — tests the router against known cases.
 *
 * Runs each test transcript through the REAL router + MLX expert server,
 * then checks whether the correct skills fired (and incorrect ones didn't).
 *
 * Usage:
 *   bun run src/listen/skills/eval.ts                    # run all cases
 *   bun run src/listen/skills/eval.ts --filter "music"   # run matching cases
 *   bun run src/listen/skills/eval.ts --concurrency 3    # parallel calls
 *
 * Requires the expert server running: cd experts && uv run serve
 */

import { readFile } from "fs/promises";
import { resolve } from "path";
import { parseArgs } from "util";
import { SkillRegistry } from "./registry";
import { classifyTranscript } from "./classify";
import { DEFAULT_SKILLS } from "./index";
import type { RouterContext, RouterResult } from "./types";
import type { ListenConfig } from "../config";
import { DEFAULT_CONFIG } from "../config";

// ── Types ─────────────────────────────────────────────────────────

interface ExpectedSkill {
  skill: string;
  action: string;
}

interface EvalExpectation {
  skills: ExpectedSkill[];
  noSkills?: string[];
  interestMin?: number;
  interestMax?: number;
}

interface EvalCase {
  name: string;
  transcript: string;
  buffer?: string;
  expect: EvalExpectation;
}

interface EvalResult {
  case: EvalCase;
  result: RouterResult;
  passed: boolean;
  failures: string[];
  latencyMs: number;
}

// ── CLI args ──────────────────────────────────────────────────────

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    filter: { type: "string", default: "" },
    concurrency: { type: "string", default: "1" },
    "cases-file": { type: "string", default: "" },
    endpoint: { type: "string", default: "http://localhost:8234" },
    verbose: { type: "boolean", default: false },
  },
  strict: true,
});

const FILTER = values.filter as string;
const CONCURRENCY = Math.max(1, Number(values.concurrency) || 1);
const VERBOSE = values.verbose as boolean;

// ── Load cases ────────────────────────────────────────────────────

async function loadCases(): Promise<EvalCase[]> {
  const file = (values["cases-file"] as string) || resolve(import.meta.dir, "eval-cases.json");
  const raw = await readFile(file, "utf-8");
  const data = JSON.parse(raw);
  let cases: EvalCase[] = data.cases || [];

  if (FILTER) {
    const f = FILTER.toLowerCase();
    cases = cases.filter((c) => c.name.toLowerCase().includes(f));
  }

  return cases;
}

// ── Evaluate a single case ────────────────────────────────────────

function checkCase(evalCase: EvalCase, result: RouterResult): string[] {
  const failures: string[] = [];
  const { expect } = evalCase;
  const got = result.matches;

  // Check expected skills are present
  for (const exp of expect.skills) {
    const found = got.some(
      (m) => m.skill === exp.skill && m.action === exp.action
    );
    if (!found) {
      const gotList = got.map((m) => `${m.skill}.${m.action}`).join(", ") || "(none)";
      failures.push(`MISSING ${exp.skill}.${exp.action} — got: ${gotList}`);
    }
  }

  // Check no-fire constraints
  if (expect.noSkills) {
    for (const forbidden of expect.noSkills) {
      const fired = got.find((m) => m.skill === forbidden);
      if (fired) {
        failures.push(
          `FALSE POSITIVE: ${fired.skill}.${fired.action} should not have fired`
        );
      }
    }
  }

  // If expect.skills is empty and no noSkills specified, check nothing fired
  if (expect.skills.length === 0 && !expect.noSkills) {
    if (got.length > 0) {
      const gotList = got.map((m) => `${m.skill}.${m.action}`).join(", ");
      failures.push(`EXPECTED no skills, got: ${gotList}`);
    }
  }

  // Check interest bounds
  if (expect.interestMin !== undefined && result.interest < expect.interestMin) {
    failures.push(
      `INTEREST too low: ${result.interest} < ${expect.interestMin}`
    );
  }
  if (expect.interestMax !== undefined && result.interest > expect.interestMax) {
    failures.push(
      `INTEREST too high: ${result.interest} > ${expect.interestMax}`
    );
  }

  return failures;
}

// ── Run eval ──────────────────────────────────────────────────────

async function runEval(
  evalCase: EvalCase,
  registry: SkillRegistry,
  config: ListenConfig
): Promise<EvalResult> {
  const ctx: RouterContext = {
    transcript: evalCase.transcript,
    buffer: evalCase.buffer || "",
    timestamp: new Date(),
  };

  const t0 = performance.now();
  const result = await classifyTranscript(ctx, registry, config);
  const latencyMs = Math.round(performance.now() - t0);

  const failures = checkCase(evalCase, result);

  return {
    case: evalCase,
    result,
    passed: failures.length === 0,
    failures,
    latencyMs,
  };
}

// ── Batch runner with concurrency control ─────────────────────────

async function runBatch(
  cases: EvalCase[],
  registry: SkillRegistry,
  config: ListenConfig
): Promise<EvalResult[]> {
  const results: EvalResult[] = [];
  const queue = [...cases];
  let completed = 0;

  async function worker() {
    while (queue.length > 0) {
      const evalCase = queue.shift()!;
      const result = await runEval(evalCase, registry, config);
      results.push(result);
      completed++;

      // Live progress
      const icon = result.passed ? "✓" : "✗";
      const color = result.passed ? "\x1b[32m" : "\x1b[31m";
      const reset = "\x1b[0m";
      const skills = result.result.matches
        .map((m) => `${m.skill}.${m.action}`)
        .join(", ") || "—";

      process.stdout.write(
        `  ${color}${icon}${reset} [${completed}/${cases.length}] ${result.case.name}` +
        `  (${result.latencyMs}ms)` +
        `  interest=${result.result.interest}` +
        `  skills=[${skills}]`
      );

      if (!result.passed) {
        process.stdout.write(`\n`);
        for (const f of result.failures) {
          console.log(`    ${color}↳ ${f}${reset}`);
        }
      } else {
        process.stdout.write(`\n`);
      }

      if (VERBOSE && result.result.reason) {
        console.log(`    reason: ${result.result.reason}`);
      }
    }
  }

  // Launch workers
  const workers = Array.from({ length: CONCURRENCY }, () => worker());
  await Promise.all(workers);

  return results;
}

// ── Report ────────────────────────────────────────────────────────

function printReport(results: EvalResult[]) {
  const passed = results.filter((r) => r.passed).length;
  const failed = results.filter((r) => !r.passed).length;
  const total = results.length;
  const avgMs = Math.round(
    results.reduce((s, r) => s + r.latencyMs, 0) / total
  );
  const totalMs = results.reduce((s, r) => s + r.latencyMs, 0);

  // Categorize
  const truePositives = results.filter(
    (r) => r.case.expect.skills.length > 0
  );
  const trueNegatives = results.filter(
    (r) => r.case.expect.skills.length === 0
  );
  const dualCases = results.filter(
    (r) => r.case.expect.skills.length > 1
  );

  const tpPassed = truePositives.filter((r) => r.passed).length;
  const tnPassed = trueNegatives.filter((r) => r.passed).length;
  const dualPassed = dualCases.filter((r) => r.passed).length;

  const passRate = ((passed / total) * 100).toFixed(0);
  const bar = passed === total ? "\x1b[32m" : "\x1b[33m";
  const reset = "\x1b[0m";

  console.log(`
  ┌─────────────────────────────────────────────┐
  │    skill classifier eval (parallel MLX)      │
  ├─────────────────────────────────────────────┤
  │  ${bar}${passed}/${total} passed (${passRate}%)${reset}${" ".repeat(Math.max(0, 29 - `${passed}/${total} passed (${passRate}%)`.length))}│
  │  true positives:  ${tpPassed}/${truePositives.length}${" ".repeat(Math.max(0, 25 - `${tpPassed}/${truePositives.length}`.length))}│
  │  true negatives:  ${tnPassed}/${trueNegatives.length}${" ".repeat(Math.max(0, 25 - `${tnPassed}/${trueNegatives.length}`.length))}│
  │  dual activation: ${dualPassed}/${dualCases.length}${" ".repeat(Math.max(0, 25 - `${dualPassed}/${dualCases.length}`.length))}│
  │  avg latency:     ${avgMs}ms${" ".repeat(Math.max(0, 25 - `${avgMs}ms`.length))}│
  │  total time:      ${(totalMs / 1000).toFixed(1)}s${" ".repeat(Math.max(0, 25 - `${(totalMs / 1000).toFixed(1)}s`.length))}│
  └─────────────────────────────────────────────┘`);

  if (failed > 0) {
    console.log(`\n  Failed cases:`);
    for (const r of results.filter((r) => !r.passed)) {
      console.log(`    \x1b[31m✗\x1b[0m ${r.case.name}`);
      for (const f of r.failures) {
        console.log(`      ↳ ${f}`);
      }
    }
  }
}

// ── Main ──────────────────────────────────────────────────────────

async function main() {
  const expertEndpoint = values.endpoint as string;

  console.log(`\n  🧪 skill classifier eval (MLX experts, parallel)`);
  console.log(`  endpoint:    ${expertEndpoint}`);
  console.log(`  concurrency: ${CONCURRENCY}`);
  if (FILTER) console.log(`  filter:      "${FILTER}"`);
  console.log();

  // Check expert server is reachable
  try {
    const res = await fetch(`${expertEndpoint}/health`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) throw new Error(`${res.status}`);
    const health = (await res.json()) as Record<string, unknown>;
    console.log(`  ✓ expert server: ${health.skills} skills loaded\n`);
  } catch {
    console.error(`  ✗ expert server not reachable at ${expertEndpoint}`);
    console.error(`  Start it: cd experts && uv run serve`);
    process.exit(1);
  }

  // Load cases
  const cases = await loadCases();
  if (cases.length === 0) {
    console.log("  No eval cases found.");
    process.exit(0);
  }
  console.log(`  running ${cases.length} cases...\n`);

  // Set up registry (same as real pipeline, but no init — skip side effects)
  const registry = new SkillRegistry();
  for (const skill of DEFAULT_SKILLS) {
    // Register without init to avoid Apple Music/watchlist side effects
    registry["skills"].set(skill.name, skill);
  }

  // Config pointing at expert server
  const config: ListenConfig = {
    ...DEFAULT_CONFIG,
    expertEndpoint,
  };

  // Run
  const results = await runBatch(cases, registry, config);

  // Report
  printReport(results);

  // Exit code
  const failed = results.filter((r) => !r.passed).length;
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("  ✗ fatal:", err);
  process.exit(1);
});
