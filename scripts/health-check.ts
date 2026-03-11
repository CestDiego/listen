#!/usr/bin/env bun
/**
 * health-check.ts — deterministic codebase health checks.
 *
 * Validates internal consistency across the listen project:
 *   - Tests pass
 *   - Python ↔ TypeScript config alignment (skills, tools, actions)
 *   - Eval cases reference only valid skill.action pairs
 *   - No stale naming references from past migrations
 *   - Knowledge base docs reference current tool/dimension names
 *   - No orphaned exports or dead skill registrations
 *
 * Exit code 0 = all checks pass, 1 = failures found.
 *
 * Usage:
 *   bun run scripts/health-check.ts          # full check
 *   bun run scripts/health-check.ts --quick  # skip tests (faster)
 *   bun run health-check                     # via package.json script
 */

import { $ } from "bun";
import { readdir } from "fs/promises";
import { join, resolve } from "path";

// ── Types ──────────────────────────────────────────────────────────

interface CheckResult {
  name: string;
  status: "pass" | "fail" | "warn";
  details?: string;
}

const ROOT = resolve(import.meta.dir, "..");
const results: CheckResult[] = [];
const skipTests = process.argv.includes("--quick");

function pass(name: string, details?: string) {
  results.push({ name, status: "pass", details });
}
function fail(name: string, details: string) {
  results.push({ name, status: "fail", details });
}
function warn(name: string, details: string) {
  results.push({ name, status: "warn", details });
}

// ── Helpers ────────────────────────────────────────────────────────

async function readJson(path: string): Promise<any> {
  return JSON.parse(await Bun.file(join(ROOT, path)).text());
}

async function readText(path: string): Promise<string> {
  return Bun.file(join(ROOT, path)).text();
}

/** Grep for a pattern across files matching a glob, return matching lines. */
async function grep(
  pattern: string,
  globs: string[],
): Promise<{ file: string; line: number; text: string }[]> {
  const matches: { file: string; line: number; text: string }[] = [];
  const regex = new RegExp(pattern, "gi");

  for (const glob of globs) {
    const globber = new Bun.Glob(glob);
    for await (const path of globber.scan({ cwd: ROOT, absolute: true })) {
      try {
        const content = await Bun.file(path).text();
        const lines = content.split("\n");
        for (let i = 0; i < lines.length; i++) {
          if (regex.test(lines[i])) {
            matches.push({
              file: path.replace(ROOT + "/", ""),
              line: i + 1,
              text: lines[i].trim(),
            });
          }
          regex.lastIndex = 0; // reset for global regex
        }
      } catch {
        // skip binary / unreadable files
      }
    }
  }
  return matches;
}

// ── Check 1: Tests ─────────────────────────────────────────────────

async function checkTests() {
  if (skipTests) {
    warn("tests", "skipped (--quick mode)");
    return;
  }

  try {
    const result = await $`bun test 2>&1`.text();
    const passMatch = result.match(/(\d+) pass/);
    const failMatch = result.match(/(\d+) fail/);
    const passes = passMatch ? parseInt(passMatch[1]) : 0;
    const failures = failMatch ? parseInt(failMatch[1]) : 0;

    if (failures > 0) {
      fail("tests", `${failures} test(s) failed (${passes} passed)`);
    } else if (passes > 0) {
      pass("tests", `${passes} tests pass`);
    } else {
      warn("tests", "no tests found");
    }
  } catch (e) {
    fail("tests", `test runner error: ${e}`);
  }
}

// ── Check 2: Config consistency (Python ↔ TS) ─────────────────────

async function checkConfigConsistency() {
  // Parse Python config.py for SKILLS and TOOL_DEFINITIONS
  const configPy = await readText("experts/experts/config.py");

  // Extract skill actions from SKILLS dict
  const skillActions = new Map<string, string[]>();
  const skillRegex = /"(\w+)":\s*\{\s*"actions":\s*\[([^\]]+)\]/g;
  let match;
  while ((match = skillRegex.exec(configPy)) !== null) {
    const skill = match[1];
    const actions = match[2]
      .match(/"(\w+)"/g)
      ?.map((s) => s.replace(/"/g, "")) ?? [];
    skillActions.set(skill, actions);
  }

  // Extract tool names from TOOL_DEFINITIONS
  const toolNames: string[] = [];
  const toolRegex = /"name":\s*"([^"]+)"/g;
  while ((match = toolRegex.exec(configPy)) !== null) {
    toolNames.push(match[1]);
  }

  // Build expected tools from SKILLS
  const expectedTools: string[] = [];
  for (const [skill, actions] of skillActions) {
    for (const action of actions) {
      expectedTools.push(`${skill}.${action}`);
    }
  }

  // Check: every expected tool has a TOOL_DEFINITION
  const missingDefs = expectedTools.filter((t) => !toolNames.includes(t));
  if (missingDefs.length > 0) {
    fail(
      "config: SKILLS→TOOL_DEFINITIONS",
      `Missing TOOL_DEFINITIONS for: ${missingDefs.join(", ")}`,
    );
  } else {
    pass(
      "config: SKILLS→TOOL_DEFINITIONS",
      `${expectedTools.length} tools defined, all have definitions`,
    );
  }

  // Check: no extra TOOL_DEFINITIONS beyond what SKILLS declares
  const extraDefs = toolNames.filter((t) => !expectedTools.includes(t));
  if (extraDefs.length > 0) {
    warn(
      "config: extra TOOL_DEFINITIONS",
      `Tools in TOOL_DEFINITIONS not in SKILLS: ${extraDefs.join(", ")}`,
    );
  }

  // Check: eval cases only reference valid skill.action pairs
  const evalCases = await readJson("src/listen/skills/eval-cases.json");
  const invalidEvalActions: string[] = [];
  for (const c of evalCases.cases ?? []) {
    if (!c.expect?.skills) continue;
    for (const s of c.expect.skills) {
      const tool = `${s.skill}.${s.action}`;
      if (!expectedTools.includes(tool)) {
        invalidEvalActions.push(`${c.name}: ${tool}`);
      }
    }
  }
  if (invalidEvalActions.length > 0) {
    fail(
      "eval-cases: valid actions",
      `Invalid skill.action in eval cases:\n  ${invalidEvalActions.join("\n  ")}`,
    );
  } else {
    const caseCount = (evalCases.cases ?? []).filter(
      (c: any) => c.expect?.skills,
    ).length;
    pass("eval-cases: valid actions", `${caseCount} cases, all reference valid tools`);
  }

  // Check: TS skill definition actions match Python config
  // Parse actions from the switch statement in handle() — the definitive list
  // of actions the skill actually handles at runtime.
  const accommodatorTs = await readText("src/listen/skills/accommodator.ts");
  const tsActions: string[] = [];
  const caseRegex = /case\s+"(\w+)":\s*\{/g;
  while ((match = caseRegex.exec(accommodatorTs)) !== null) {
    tsActions.push(match[1]);
  }

  const pyAccActions = skillActions.get("accommodator") ?? [];
  const missingInTs = pyAccActions.filter((a) => !tsActions.includes(a));
  const extraInTs = tsActions.filter((a) => !pyAccActions.includes(a));

  if (missingInTs.length > 0 || extraInTs.length > 0) {
    const parts: string[] = [];
    if (missingInTs.length) parts.push(`missing in TS: ${missingInTs.join(", ")}`);
    if (extraInTs.length) parts.push(`extra in TS: ${extraInTs.join(", ")}`);
    fail("config: TS↔Python action alignment", parts.join("; "));
  } else {
    pass(
      "config: TS↔Python action alignment",
      `${pyAccActions.length} accommodator actions match`,
    );
  }

  // Check: DEFAULT_SKILLS array contents (look at the actual assignment, not comments)
  const skillsIndex = await readText("src/listen/skills/index.ts");
  const defaultSkillsMatch = skillsIndex.match(
    /export const DEFAULT_SKILLS[^=]*=\s*\[([^\]]+)\]/,
  );
  const defaultSkillsValue = defaultSkillsMatch?.[1] ?? "";

  if (defaultSkillsValue.includes("musicSkill")) {
    fail("skills: DEFAULT_SKILLS", "musicSkill is still in DEFAULT_SKILLS array");
  } else {
    pass("skills: DEFAULT_SKILLS", "musicSkill correctly excluded");
  }

  if (defaultSkillsValue.includes("accommodatorSkill")) {
    pass("skills: DEFAULT_SKILLS", "accommodatorSkill registered");
  } else {
    fail("skills: DEFAULT_SKILLS", "accommodatorSkill not found in DEFAULT_SKILLS");
  }
}

// ── Check 3: Stale naming references ───────────────────────────────

async function checkStaleRefs() {
  // Define retired names and where they should NOT appear
  const stalePatterns: {
    name: string;
    pattern: string;
    globs: string[];
    /** Files/paths to exclude (legacy files, migration code) */
    exclude: string[];
  }[] = [
    {
      name: "music.play/pause/skip/etc in runtime code",
      pattern: "\\bmusic\\.(play|pause|resume|skip|previous|volume_up|volume_down|dislike)\\b",
      globs: [
        "src/**/*.ts",
        "experts/experts/serve_multitool.py",
        "experts/experts/evaluate_multitool.py",
      ],
      exclude: [
        // Legacy skill file — expected to reference music.*
        "src/listen/skills/music.ts",
        // Generator intentionally maps old music templates
        "experts/experts/generate_multitool.py",
        "experts/experts/generate.py",
        // Legacy per-skill files
        "experts/experts/train.py",
        "experts/experts/evaluate.py",
        "experts/experts/serve.py",
      ],
    },
    {
      name: '"music" as dimension key',
      pattern: '(key:\\s*"music"|dimensions\\.music|\\[.music.\\].*dimension)',
      globs: ["src/listen/intent-vector.ts", "src/listen/skills/classify.ts", "src/listen/skills/router.ts"],
      exclude: [],
    },
    {
      name: "stale music.* in knowledge base",
      pattern: "\\bmusic\\.(play|pause|resume|skip|previous|volume_up|volume_down)\\b",
      globs: [".knowledge/*.md"],
      exclude: [],
    },
  ];

  for (const check of stalePatterns) {
    const matches = await grep(check.pattern, check.globs);
    const filtered = matches.filter(
      (m) => !check.exclude.some((ex) => m.file.includes(ex)),
    );

    if (filtered.length > 0) {
      const lines = filtered
        .slice(0, 5)
        .map((m) => `  ${m.file}:${m.line}: ${m.text.slice(0, 80)}`);
      const extra = filtered.length > 5 ? `\n  ...and ${filtered.length - 5} more` : "";
      fail(
        `stale refs: ${check.name}`,
        `${filtered.length} match(es):\n${lines.join("\n")}${extra}`,
      );
    } else {
      pass(`stale refs: ${check.name}`);
    }
  }
}

// ── Check 4: Knowledge base freshness ──────────────────────────────

async function checkKnowledgeBase() {
  const knowledgeDir = join(ROOT, ".knowledge");
  try {
    await readdir(knowledgeDir);
  } catch {
    warn("knowledge base", ".knowledge/ directory not found");
    return;
  }

  // Check that dimension table in architecture.md matches DIMENSION_DEFS
  const arch = await readText(".knowledge/architecture.md");
  const intentVector = await readText("src/listen/intent-vector.ts");

  // Extract dimension keys from DIMENSION_DEFS
  const dimKeys: string[] = [];
  const dimRegex = /key:\s*"(\w+)"/g;
  let match;
  while ((match = dimRegex.exec(intentVector)) !== null) {
    dimKeys.push(match[1]);
  }

  // Check each dimension key appears in architecture.md dimensions table
  const missingInDocs = dimKeys.filter(
    (k) => !arch.includes(`\`${k}\``) && !arch.includes(`| ${k} `),
  );
  if (missingInDocs.length > 0) {
    warn(
      "knowledge: dimension docs",
      `Dimensions missing from architecture.md: ${missingInDocs.join(", ")}`,
    );
  } else {
    pass("knowledge: dimension docs", `all ${dimKeys.length} dimensions documented`);
  }

  // Check tool definitions section mentions current tools
  const configPy = await readText("experts/experts/config.py");
  const toolNames: string[] = [];
  const toolRegex = /"name":\s*"([^"]+)"/g;
  while ((match = toolRegex.exec(configPy)) !== null) {
    toolNames.push(match[1]);
  }

  const missingToolDocs = toolNames.filter((t) => !arch.includes(t));
  if (missingToolDocs.length > 0) {
    warn(
      "knowledge: tool docs",
      `Tools missing from architecture.md: ${missingToolDocs.join(", ")}`,
    );
  } else {
    pass("knowledge: tool docs", `all ${toolNames.length} tools documented`);
  }
}

// ── Check 5: Playlist infrastructure ───────────────────────────────

async function checkPlaylistDirs() {
  const quadrants = ["uplift", "release", "calm", "comfort", "focus", "neutral"];
  const playlistDir = join(ROOT, "data", "playlists");

  const missing: string[] = [];
  for (const q of quadrants) {
    const metaPath = join(playlistDir, q, "_meta.json");
    const exists = await Bun.file(metaPath).exists();
    if (!exists) missing.push(q);
  }

  if (missing.length > 0) {
    warn("playlists: quadrant dirs", `Missing _meta.json for: ${missing.join(", ")}`);
  } else {
    pass("playlists: quadrant dirs", `all ${quadrants.length} quadrants have _meta.json`);
  }
}

// ── Check 6: Training data system prompt matches config ────────────

async function checkTrainingData() {
  const trainPath = join(ROOT, "experts/data/multitool/train.jsonl");
  const exists = await Bun.file(trainPath).exists();
  if (!exists) {
    warn("training data", "experts/data/multitool/train.jsonl not found");
    return;
  }

  // Read first entry and extract system prompt tools
  const firstLine = (await Bun.file(trainPath).text()).split("\n")[0];
  const entry = JSON.parse(firstLine);
  const systemPrompt: string = entry.messages[0].content;

  // Extract tool names from config.py
  const configPy = await readText("experts/experts/config.py");
  const toolNames: string[] = [];
  const toolRegex = /"name":\s*"([^"]+)"/g;
  let match;
  while ((match = toolRegex.exec(configPy)) !== null) {
    toolNames.push(match[1]);
  }

  // Check each config tool appears in training data system prompt
  const missingInTraining = toolNames.filter((t) => !systemPrompt.includes(t));
  if (missingInTraining.length > 0) {
    fail(
      "training data: system prompt",
      `Tools in config.py but missing from training data system prompt: ${missingInTraining.join(", ")}. Run: uv run python -m experts.generate_multitool`,
    );
  } else {
    pass(
      "training data: system prompt",
      `all ${toolNames.length} tools present in training data`,
    );
  }
}

// ── Runner ─────────────────────────────────────────────────────────

async function main() {
  console.log("\n  ┌────────────────────────────────────────┐");
  console.log("  │  🩺 listen — codebase health check     │");
  console.log("  └────────────────────────────────────────┘\n");

  await checkTests();
  await checkConfigConsistency();
  await checkStaleRefs();
  await checkKnowledgeBase();
  await checkPlaylistDirs();
  await checkTrainingData();

  // ── Report ──────────────────────────────────────────────────────

  console.log("");
  const icon = { pass: "  ✅", fail: "  ❌", warn: "  ⚠️ " };
  const failures: CheckResult[] = [];
  const warnings: CheckResult[] = [];

  for (const r of results) {
    const detail = r.details ? ` — ${r.details}` : "";
    console.log(`${icon[r.status]} ${r.name}${detail}`);
    if (r.status === "fail") failures.push(r);
    if (r.status === "warn") warnings.push(r);
  }

  const total = results.length;
  const passes = results.filter((r) => r.status === "pass").length;

  console.log("\n  ─────────────────────────────────────────");
  console.log(
    `  ${passes}/${total} checks pass` +
      (warnings.length ? `, ${warnings.length} warning(s)` : "") +
      (failures.length ? `, ${failures.length} failure(s)` : ""),
  );

  if (failures.length > 0) {
    console.log("  ❌ Health check FAILED\n");
    process.exit(1);
  } else if (warnings.length > 0) {
    console.log("  ⚠️  Health check passed with warnings\n");
  } else {
    console.log("  ✅ All clear\n");
  }
}

main().catch((err) => {
  console.error("  ✗ health check error:", err);
  process.exit(1);
});
