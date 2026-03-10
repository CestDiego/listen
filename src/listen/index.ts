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
import { IntentVectorStore, ActivationGate, type GateResult } from "./intent-vector";
import { startDashboard, type TranscriptPost } from "./dashboard";
import {
  SkillRegistry,
  classifyTranscript,
  DEFAULT_SKILLS,
  type RouterContext,
  type ClassifyResult,
  type SkillExecution,
} from "./skills";
import type { DecisionSkillMatch, DecisionExpertResult } from "./session";

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

let shuttingDown = false;

async function shutdown(signal: string): Promise<void> {
  if (shuttingDown) return; // ignore repeated signals
  shuttingDown = true;
  running = false;

  console.log(`\n  ⏹  stopping listen... (${signal})`);
  if (activeSession) {
    await activeSession.save();
    console.log("  💾 session saved.");
  }
  process.exit(0);
}

process.on("SIGINT", () => { shutdown("SIGINT"); });
process.on("SIGTERM", () => { shutdown("SIGTERM"); });

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

  // Intent vector engine (multi-dimensional activation tracking)
  const intentVector = new IntentVectorStore();
  const activationGate = new ActivationGate();

  // Web dashboard (SSE live updates + optional transcript POST handler)
  startDashboard(session, onTranscript);

  return { registry, events, session, intentVector, activationGate };
}

/**
 * Process a transcript through the skill classifier.
 *
 * Context flow:
 *   1. Build context with transcript + buffer + recent skill history
 *   2. Classify — parallel fan-out to all skill experts via Promise.all
 *   3. Log per-expert observability (latency, status, confidence)
 *   4. Enrich context so handlers see the full picture
 *   5. Execute matched skills — each handler sees why it was triggered + siblings
 *   6. Record executions into skill history (fed back to future classifier calls)
 *   7. If interest is high → build situation summary → analyzer sees everything
 */
async function processTranscript(
  text: string,
  buffer: TranscriptBuffer,
  entryId: string,
  registry: SkillRegistry,
  events: EventEmitter,
  session: SessionStore,
  intentVector: IntentVectorStore,
  activationGate: ActivationGate,
  config: ListenConfig,
  cycleLabel: string
): Promise<void> {
  if (!text || buffer.wordCount < 3) return;

  // 1. Build context — includes recent skill history so classifier avoids re-triggers
  const ctx: RouterContext = {
    transcript: text,
    buffer: buffer.recentText(2),
    timestamp: new Date(),
    recentSkills: recentSkillExecutions.slice(-10),
  };

  // 2. Capture skill state BEFORE classification (for observability)
  const skillState = await registry.buildStateContext();

  // 3. Classify — parallel fan-out to all skill experts
  const result: ClassifyResult = await classifyTranscript(ctx, registry, config);

  // 4. Enrich context so skill handlers see the full picture
  ctx.routerReason = result.reason;
  ctx.allMatches = result.matches;

  const matchedNames = result.matches.map((m) => `${m.skill}.${m.action}`);
  const escalated = result.interest >= config.threshold;

  // 5. Build per-expert observability records for session
  const expertResults: DecisionExpertResult[] = result.experts.map((e) => ({
    skill: e.skill,
    match: e.match,
    action: e.action,
    confidence: e.confidence,
    expertMs: e.expertMs,
    roundTripMs: e.roundTripMs,
    status: e.status,
    error: e.error,
  }));

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

  // 5b. Update intent vector from classify results
  const vectorInputs = result.experts.map(e => ({
    skill: e.skill,
    match: e.match,
    confidence: e.confidence,
  }));
  const vectorSnapshot = intentVector.update(vectorInputs);

  // 5c. Activation gate — post-classification wellbeing bias (hysteresis)
  const classifierMatchedWellbeing = result.matches.some(m => m.skill === "wellbeing");
  const classifierWellbeingConf = result.experts.find(e => e.skill === "wellbeing")?.confidence ?? 0;
  const gateResult = activationGate.evaluate(
    classifierMatchedWellbeing,
    classifierWellbeingConf,
    vectorSnapshot.dimensions.wellbeing,
  );

  // If gate promotes, inject a synthetic wellbeing match
  if (gateResult.promoted && gateResult.promotedConfidence) {
    result.matches.push({
      skill: "wellbeing",
      action: "check_in",
      params: {},
      confidence: gateResult.promotedConfidence,
    });
    result.interest = Math.max(result.interest, 7); // Bump interest for promoted wellbeing
    result.reason += ` [gate:promoted@${gateResult.promotedConfidence.toFixed(2)}]`;
    if (config.verbose) {
      console.log(`  [${cycleLabel}] 🛡️ gate promoted wellbeing (state=${gateResult.state}, level=${gateResult.wellbeingLevel.toFixed(2)}, conf=${gateResult.promotedConfidence.toFixed(2)})`);
    }
  } else if (gateResult.state !== "idle" && config.verbose) {
    console.log(`  [${cycleLabel}] 🛡️ gate: ${gateResult.state} (level=${gateResult.wellbeingLevel.toFixed(2)}, threshold=${gateResult.effectiveThreshold})`);
  }

  session.emitIntentVector(vectorSnapshot, intentVector.history(), gateResult);

  // 6. Record the full decision with per-expert breakdown
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
    classifyMs: result.classifyMs,
    expertSumMs: result.expertSumMs,
    expertResults,
    escalated,
    wordCount: buffer.wordCount,
    latencyMs: result.classifyMs, // backward compat
  });

  // 7. Log to JSONL — both the legacy router event and the new classify fanout
  await events.classifyFanout({
    classifyMs: result.classifyMs,
    expertSumMs: result.expertSumMs,
    experts: result.experts.map((e) => ({
      skill: e.skill,
      match: e.match,
      action: e.action,
      confidence: e.confidence,
      expertMs: e.expertMs,
      roundTripMs: e.roundTripMs,
      status: e.status,
      error: e.error,
    })),
    matchedSkills: matchedNames,
    interest: result.interest,
  });

  // 8. Rich console output with per-expert breakdown
  if (config.verbose) {
    const gain = result.classifyMs > 0
      ? (result.expertSumMs / result.classifyMs).toFixed(1)
      : "1.0";
    const icon = result.matches.length > 0
      ? "⚡"
      : escalated
        ? "🟢"
        : "⚪";
    const skillList = matchedNames.length > 0
      ? ` → [${matchedNames.join(", ")}]`
      : "";

    console.log(
      `  ${cycleLabel} 🎯 ${icon} classify ${result.experts.length} experts in ${result.classifyMs}ms` +
      ` (sum: ${result.expertSumMs}ms, gain: ${gain}×)` +
      ` interest=${result.interest}/10${skillList}`
    );

    // Per-expert breakdown
    for (const e of result.experts) {
      const matchIcon = e.match ? "+" : "-";
      const action = e.match ? `.${e.action}` : "";
      const conf = e.confidence
        ? ` [${(e.confidence * 100).toFixed(0)}%]`
        : "";
      const status = e.status !== "ok" ? ` (${e.status})` : "";
      const pad = e.skill.padEnd(12);
      console.log(
        `  ${cycleLabel}   ${matchIcon} ${pad}${action}${conf} ${e.roundTripMs}ms (expert: ${e.expertMs}ms)${status}`
      );
    }
  }

  // 9. Execute matched skills — each handler sees enriched context
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

    // Record into skill history for future classifier calls
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
      true,
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

  // 10. Escalate to big model analysis if interest is high
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
  classifyResult: ClassifyResult,
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

  // What the classifier detected
  const expertLines = classifyResult.experts.map((e) => {
    const icon = e.match ? "MATCHED" : "no match";
    const action = e.match ? `.${e.action}` : "";
    const conf = e.confidence
      ? ` (${(e.confidence * 100).toFixed(0)}% confidence)`
      : "";
    return `  ${e.skill}${action}: ${icon}${conf} [${e.roundTripMs}ms]`;
  });
  sections.push(
    `CLASSIFIER ASSESSMENT:\n` +
      `  Interest: ${classifyResult.interest}/10\n` +
      `  Classification time: ${classifyResult.classifyMs}ms (parallel)\n` +
      `  Expert breakdown:\n${expertLines.join("\n")}`
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
  │  classifier    ${"MLX experts (parallel)".padEnd(29)}│
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
  const { registry, events, session, intentVector, activationGate } = await initSystems(config);

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
        intentVector,
        activationGate,
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
      intentVector,
      activationGate,
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
  const { registry, events, session, intentVector, activationGate } = await initSystems(config, serialOnTranscript);

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
  const { registry, events, session, intentVector, activationGate } = await initSystems(config);

  console.log(
    "  📋 pipe mode — paste transcript, press Enter, Ctrl+D to finish."
  );
  console.log("  Skill classifier runs on every input (parallel fan-out).\n");

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
      intentVector,
      activationGate,
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
      intentVector,
      activationGate,
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
