#!/usr/bin/env bun
/**
 * listen — user state observability CLI
 *
 * Records audio → transcribes locally → routes through skill system →
 * skills take actions (music, wellbeing, etc.) → emits events →
 * escalates high-interest content to big model for analysis.
 *
 * Usage:
 *   bun listen              # live mic mode
 *   bun listen --pipe       # paste/pipe transcripts via stdin
 *   bun listen --moonshine  # receive from Moonshine menu bar app
 *
 * Events:
 *   tail -f /tmp/listen-events.jsonl   # consume from another process
 */

import {
  parseCliArgs,
  type ListenConfig,
  type TranscriptChunk,
} from "./config";
import { TranscriptBuffer } from "./buffer";
import { initRecorder, recordChunk, cleanupChunk } from "./recorder";
import { transcribe } from "./transcriber";
import { analyze } from "./analyzer";
import { notify } from "./notifier";
import { EventEmitter } from "./events";
import { respondToSkill } from "./responder";
import { SessionStore } from "./session";
import { startDashboard, type TranscriptPost } from "./dashboard";
import {
  SkillRegistry,
  routeTranscript,
  DEFAULT_SKILLS,
  type RouterContext,
  type RouterResult,
  type SkillExecution,
} from "./skills";
import type { DecisionSkillMatch } from "./session";

// ── State ──────────────────────────────────────────────────────────

let running = true;
let activeSession: SessionStore | null = null;

/** Rolling log of recent skill executions — fed back to router + analyzer. */
const recentSkillExecutions: SkillExecution[] = [];
const MAX_SKILL_HISTORY = 20;

function recordSkillExecution(exec: SkillExecution) {
  recentSkillExecutions.push(exec);
  if (recentSkillExecutions.length > MAX_SKILL_HISTORY) {
    recentSkillExecutions.shift();
  }
}

// ── Signal handling ────────────────────────────────────────────────

process.on("SIGINT", async () => {
  console.log("\n  ⏹  stopping listen...");
  running = false;
  if (activeSession) {
    await activeSession.save();
    console.log("  💾 session saved.");
  }
});
process.on("SIGTERM", async () => {
  running = false;
  if (activeSession) {
    await activeSession.save();
  }
});

// ── Shared init ────────────────────────────────────────────────────

async function initSystems(
  config: ListenConfig,
  onTranscript?: (post: TranscriptPost) => void | Promise<void>
) {
  // Skill registry — register all built-in skills
  const registry = new SkillRegistry();
  for (const skill of DEFAULT_SKILLS) {
    await registry.register(skill);
  }
  console.log(`  🧩 skills: ${registry.summary()}`);

  // Event emitter (JSONL log for external agents)
  const events = new EventEmitter(config.eventLogPath, config.verbose);
  console.log(`  📡 events → ${config.eventLogPath}`);

  // Session store (structured timeline + persistence)
  const sessionPath = `/tmp/listen-session-${Date.now()}.json`;
  const session = new SessionStore(sessionPath, config as unknown as Record<string, unknown>);
  activeSession = session;
  console.log(`  💾 session → ${sessionPath}`);

  // Web dashboard (SSE live updates + optional transcript POST handler)
  startDashboard(session, onTranscript);

  return { registry, events, session };
}

/**
 * Process a transcript through the skill router.
 * This is the unified pipeline that replaces the old gate + watchlist.
 *
 * Context flow:
 *   1. Build RouterContext with transcript + buffer + recent skill history
 *   2. Route (single LLM call) — router sees recent skills to avoid re-triggers
 *   3. Enrich context with router reasoning + all matches (so handlers see the full picture)
 *   4. Execute matched skills — each handler sees why it was triggered + siblings
 *   5. Record executions into skill history (fed back to future router calls)
 *   6. If interest is high → build situation summary → analyzer sees everything
 */
async function processTranscript(
  text: string,
  buffer: TranscriptBuffer,
  entryId: string,
  registry: SkillRegistry,
  events: EventEmitter,
  session: SessionStore,
  config: ListenConfig,
  cycleLabel: string
): Promise<void> {
  if (!text || buffer.wordCount < 3) return;

  // 1. Build context — includes recent skill history so router avoids re-triggers
  const ctx: RouterContext = {
    transcript: text,
    buffer: buffer.recentText(2),
    timestamp: new Date(),
    recentSkills: recentSkillExecutions.slice(-10), // last 10 executions
  };

  // 2. Capture skill state BEFORE routing (for observability)
  const skillState = await registry.buildStateContext();

  // 3. Route — single LLM call classifies across all skills
  const t0 = performance.now();
  const result: RouterResult = await routeTranscript(ctx, registry, config);
  const routerMs = Math.round(performance.now() - t0);

  // 4. Enrich context so skill handlers see the full picture
  ctx.routerReason = result.reason;
  ctx.allMatches = result.matches;

  const matchedNames = result.matches.map((m) => `${m.skill}.${m.action}`);
  const escalated = result.interest >= config.threshold;

  // 5. Record the full router decision for observability
  const decisionMatches: DecisionSkillMatch[] = result.matches.map((m) => ({
    skill: m.skill,
    action: m.action,
    params: m.params,
    confidence: m.confidence,
  }));

  const recentForDecision = (ctx.recentSkills || []).map((s) => ({
    skill: s.skill,
    action: s.action,
    success: s.success,
    voice: s.voice,
    agoSeconds: Math.round((Date.now() - s.timestamp.getTime()) / 1000),
  }));

  const decision = session.addRouterDecision({
    entryId,
    timestamp: new Date().toISOString(),
    transcript: text,
    bufferContext: ctx.buffer,
    skillState,
    recentSkills: recentForDecision,
    interest: result.interest,
    reason: result.reason,
    matches: decisionMatches,
    latencyMs: routerMs,
    escalated,
    wordCount: buffer.wordCount,
  });

  // Log to JSONL event file
  await events.routerResult(
    result.interest,
    result.reason,
    matchedNames,
    buffer.wordCount
  );

  if (config.verbose) {
    const icon = result.matches.length > 0
      ? "⚡"
      : escalated
        ? "🟢"
        : "⚪";
    const skillList = matchedNames.length > 0
      ? ` → [${matchedNames.join(", ")}]`
      : "";
    console.log(
      `  ${cycleLabel} 🧭 ${icon} interest=${result.interest}/10 (${routerMs}ms)${skillList} — ${result.reason}`
    );
  }

  // 6. Execute matched skills — each handler sees enriched context
  const executionResults: Array<{
    skill: string;
    action: string;
    success: boolean;
    voice?: string;
    confidence: number;
  }> = [];

  for (const match of result.matches) {
    if (!running) break;

    if (config.verbose) {
      console.log(
        `  ${cycleLabel} ⚡ ${match.skill}.${match.action}(${JSON.stringify(match.params)}) [${(match.confidence * 100).toFixed(0)}%]`
      );
    }

    const skillResult = await registry.execute(match, ctx);

    // 7. Record into skill history for future router calls
    const exec: SkillExecution = {
      skill: match.skill,
      action: match.action,
      success: skillResult.success,
      voice: skillResult.voice,
      timestamp: new Date(),
    };
    recordSkillExecution(exec);

    executionResults.push({
      skill: match.skill,
      action: match.action,
      success: skillResult.success,
      voice: skillResult.voice,
      confidence: match.confidence,
    });

    // Update the decision record with execution results
    session.updateDecisionSkill(
      decision.id,
      match.skill,
      match.action,
      true, // executed
      skillResult.success,
      skillResult.voice
    );

    // Log skill execution
    await events.skillExecuted(
      match.skill,
      match.action,
      skillResult.success,
      match.confidence
    );
    session.addSkillResult(
      match.skill,
      match.action,
      skillResult.success,
      skillResult.voice,
      entryId
    );

    // Respond (sound, voice, notification)
    if (skillResult.success) {
      await respondToSkill(match.skill, skillResult);
    }

    if (config.verbose && skillResult.success) {
      console.log(
        `  ${cycleLabel}   ✓ ${match.skill}: ${skillResult.voice || "(no voice)"}`
      );
    }
  }

  // 8. Escalate to big model analysis if interest is high
  //    Analyzer gets FULL situation context: transcript + skill actions + results
  if (escalated && running) {
    if (config.verbose) {
      console.log(
        `  ${cycleLabel} 🚀 analyzing with ${config.analysisModel}...`
      );
    }

    const situationContext = buildSituationContext(
      buffer.fullContextTimestamped(),
      result,
      executionResults
    );
    const analysisResult = await analyze(situationContext, result.reason, config);
    await events.analysisComplete(result.reason, analysisResult.insights);
    session.addAnalysis(analysisResult, entryId);
    await notify(analysisResult);
  }
}

/**
 * Build a rich situation summary for the analyzer.
 * The big model sees not just the transcript, but what the system detected
 * and what actions were taken — so it can provide genuinely useful analysis.
 */
function buildSituationContext(
  timestampedTranscript: string,
  routerResult: RouterResult,
  executions: Array<{
    skill: string;
    action: string;
    success: boolean;
    voice?: string;
    confidence: number;
  }>
): string {
  const sections: string[] = [];

  // Transcript
  sections.push(`CONVERSATION TRANSCRIPT:\n${timestampedTranscript}`);

  // What the router detected
  sections.push(
    `ROUTER ASSESSMENT:\n` +
      `  Interest: ${routerResult.interest}/10\n` +
      `  Reason: ${routerResult.reason}`
  );

  // What skills fired and what they did
  if (executions.length > 0) {
    const lines = executions.map((e) => {
      const status = e.success ? "succeeded" : "FAILED";
      const response = e.voice ? ` → responded: "${e.voice}"` : "";
      return `  ${e.skill}.${e.action} [${(e.confidence * 100).toFixed(0)}% confidence] — ${status}${response}`;
    });
    sections.push(`ACTIONS TAKEN:\n${lines.join("\n")}`);
  }

  return sections.join("\n\n");
}

// ── Banner ─────────────────────────────────────────────────────────

function printBanner(config: ListenConfig): void {
  const isMoonshine = config.mode === "moonshine";
  const transcriber = isMoonshine
    ? "moonshine (Swift app)"
    : (config.whisperModel.split("/").pop() ?? config.whisperModel);
  const audioLine = isMoonshine
    ? "Moonshine (menu bar)"
    : config.audioDevice;

  console.log(`
  ┌─────────────────────────────────────────────┐
  │       🎧 listen — skill-based pipeline      │
  ├─────────────────────────────────────────────┤
  │  mode          ${config.mode.padEnd(29)}│
  │  audio device  ${audioLine.slice(0, 29).padEnd(29)}│
  │  transcriber   ${transcriber.slice(0, 29).padEnd(29)}│
  │  router        ${"MLX experts (local)".padEnd(29)}│
  │  expert server ${config.expertEndpoint.padEnd(29)}│
  │  analysis      ${config.analysisModel.padEnd(29)}│
  │  threshold     ${(config.threshold + "/10").padEnd(29)}│
  │  buffer        ${(config.bufferMinutes + " min").padEnd(29)}│
  │  event log     ${config.eventLogPath.padEnd(29)}│
  ├─────────────────────────────────────────────┤
  │  Ctrl+C to stop                             │
  └─────────────────────────────────────────────┘
  `);
}

// ── Live mic mode ──────────────────────────────────────────────────

async function runLiveMode(config: ListenConfig): Promise<void> {
  await initRecorder(config);
  const buffer = new TranscriptBuffer(config.bufferMinutes);
  const { registry, events, session } = await initSystems(config);

  let cycle = 0;

  while (running) {
    cycle++;
    const cycleLabel = `[${String(cycle).padStart(4, "0")}]`;

    // 1. Record
    if (config.verbose) {
      process.stdout.write(
        `  ${cycleLabel} 🔴 recording ${config.chunkSeconds}s...`
      );
    }

    let audioPath: string;
    try {
      audioPath = await recordChunk(config);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`\n  ✗ recording failed: ${message}`);
      break;
    }

    if (!running) break;

    // 2. Transcribe
    if (config.verbose) process.stdout.write(" → transcribing...");

    const chunk = await transcribe(audioPath, config);
    await cleanupChunk(audioPath);

    if (chunk.text) {
      buffer.append(chunk);
      const entry = session.addChunk(cycle, chunk.text, config.chunkSeconds);

      if (config.verbose) {
        console.log(` ✓`);
        console.log(`  ${cycleLabel} 📝 "${truncate(chunk.text, 80)}"`);
      }

      // 3. Route through skill system
      await processTranscript(
        chunk.text,
        buffer,
        entry.id,
        registry,
        events,
        session,
        config,
        cycleLabel
      );
    } else {
      if (config.verbose) console.log(` (silence)`);
      session.addChunk(cycle, "", config.chunkSeconds);
    }
  }

  await session.save();
  console.log("  ✓ listen stopped cleanly.");
}

// ── Moonshine mode ─────────────────────────────────────────────────
// Receives transcript POSTs from the Moonshine Swift menu bar app.
// No ffmpeg, no whisper — the Swift app handles audio capture + transcription.
// This process runs the dashboard, skill router, and analysis pipeline.

async function runMoonshineMode(config: ListenConfig): Promise<void> {
  const buffer = new TranscriptBuffer(config.bufferMinutes);
  let cycle = 0;
  let ready = false;

  // Serial queue: ensures transcript callbacks execute one at a time,
  // preventing race conditions on cycle counter and shared buffer state.
  let pending = Promise.resolve();

  // The onTranscript callback processes each incoming line from the Swift app
  const onTranscript = async (post: TranscriptPost) => {
    if (!running || !ready) return;

    cycle++;
    const cycleLabel = `[${String(cycle).padStart(4, "0")}]`;

    const chunk: TranscriptChunk = {
      text: post.text,
      timestamp: new Date(),
      durationSeconds: post.durationSeconds,
    };

    buffer.append(chunk);
    const entry = session.addChunk(cycle, post.text, post.durationSeconds);

    if (config.verbose) {
      console.log(
        `  ${cycleLabel} 📝 [${post.source}] "${truncate(post.text, 80)}" (${post.durationSeconds.toFixed(1)}s)`
      );
    }

    // Route through skill system
    await processTranscript(
      post.text,
      buffer,
      entry.id,
      registry,
      events,
      session,
      config,
      cycleLabel
    );
  };

  // Serialized wrapper: queues transcript processing to avoid concurrent mutations
  const serialOnTranscript = (post: TranscriptPost) => {
    pending = pending
      .then(() => onTranscript(post))
      .catch((err) => console.error("  ⚠ transcript callback error:", err));
  };

  // Init systems with the serialized transcript callback
  const { registry, events, session } = await initSystems(config, serialOnTranscript);

  // Mark ready — safe to process transcripts now that all systems are initialized
  ready = true;

  console.log("  🌙 moonshine mode — waiting for transcripts from Moonshine app");
  console.log("  📋 POST transcripts to http://localhost:3838/api/transcript");
  console.log("  🖥  Open the Moonshine menu bar app to start transcribing.\n");

  // Keep the process alive until SIGINT sets running=false
  await new Promise<void>((resolve) => {
    const check = setInterval(() => {
      if (!running) {
        clearInterval(check);
        resolve();
      }
    }, 500);
  });

  await session.save();
  console.log("  ✓ moonshine mode stopped cleanly.");
}

// ── Pipe/stdin mode ────────────────────────────────────────────────

async function runPipeMode(config: ListenConfig): Promise<void> {
  const buffer = new TranscriptBuffer(config.bufferMinutes);
  const { registry, events, session } = await initSystems(config);

  console.log(
    "  📋 pipe mode — paste transcript, press Enter, Ctrl+D to finish."
  );
  console.log("  Skill router runs on every input.\n");

  const decoder = new TextDecoder();
  let cycle = 0;

  for await (const raw of Bun.stdin.stream()) {
    if (!running) break;

    const text = decoder.decode(raw, { stream: true }).trim();
    if (!text) continue;

    cycle++;
    const chunk: TranscriptChunk = {
      text,
      timestamp: new Date(),
      durationSeconds: 0,
    };
    buffer.append(chunk);
    const entry = session.addChunk(cycle, text, 0);

    if (config.verbose) {
      console.log(
        `  📝 +${text.split(/\s+/).length} words (${buffer.wordCount} total)`
      );
    }

    // Route through skill system
    await processTranscript(
      text,
      buffer,
      entry.id,
      registry,
      events,
      session,
      config,
      "     "
    );
  }

  // Final check on remaining buffer
  if (buffer.wordCount > 20) {
    console.log(`\n  🧭 final router check on remaining buffer...`);
    await processTranscript(
      buffer.fullContext(),
      buffer,
      session.getTimeline().at(-1)?.id ?? "final",
      registry,
      events,
      session,
      config,
      "     "
    );
  }

  await session.save();
  console.log("  ✓ pipe mode finished.");
}

// ── Main ───────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const config = parseCliArgs();
  printBanner(config);

  if (config.mode === "moonshine") {
    await runMoonshineMode(config);
  } else if (config.mode === "pipe") {
    await runPipeMode(config);
  } else {
    await runLiveMode(config);
  }
}

// ── Util ───────────────────────────────────────────────────────────

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + "…" : s;
}

// ── Run ────────────────────────────────────────────────────────────

main().catch((err) => {
  console.error("  ✗ fatal:", err);
  process.exit(1);
});
