#!/usr/bin/env bun
/**
 * listen — user state observability CLI
 *
 * Records audio → transcribes locally → checks watchlist patterns →
 * gates with cheap model → analyzes with big model →
 * emits structured events → responds (sound, voice, notification).
 *
 * Usage:
 *   bun listen              # live mic mode
 *   bun listen --pipe       # paste/pipe transcripts via stdin
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
import { runGate } from "./gate";
import { analyze } from "./analyzer";
import { notify } from "./notifier";
import { WatchlistMatcher } from "./watchlist";
import { EventEmitter } from "./events";
import { respond } from "./responder";
import { SessionStore } from "./session";
import { startDashboard, type TranscriptPost } from "./dashboard";

// ── State ──────────────────────────────────────────────────────────

let running = true;
let activeSession: SessionStore | null = null;

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
  // Watchlist
  const watchlist = new WatchlistMatcher();
  if (config.watchlistPath) {
    const count = await watchlist.load(config.watchlistPath);
    if (count > 0) {
      const summary = watchlist.summary();
      const cats = Object.entries(summary)
        .map(([k, v]) => `${k}(${v})`)
        .join(" ");
      console.log(`  📋 watchlist: ${count} patterns loaded [${cats}]`);
    }
  }

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

  return { watchlist, events, session };
}

/**
 * Run watchlist check on a chunk. Fires IMMEDIATELY (no gating).
 * This is the fast path — pure string matching, no LLM calls.
 */
async function checkWatchlist(
  text: string,
  context: string,
  entryId: string,
  watchlist: WatchlistMatcher,
  events: EventEmitter,
  session: SessionStore,
  config: ListenConfig,
  cycleLabel: string
): Promise<void> {
  if (!text) return;

  const matches = watchlist.check(text);

  for (const match of matches) {
    if (config.verbose) {
      console.log(
        `  ${cycleLabel} 🫀 WATCHLIST HIT [${match.pattern.severity}] ${match.pattern.category}/${match.pattern.id} → "${match.trigger}"`
      );
    }

    // 1. Emit event (for external agents)
    await events.watchlistMatch(
      match.pattern.id,
      match.pattern.category,
      match.pattern.severity,
      match.trigger,
      match.matchedText,
      context
    );

    // 2. Record in session (for dashboard)
    session.addWatchlistMatch(match, entryId);

    // 3. Execute response (sound → voice → notification)
    await respond(match);
  }
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
  │         🎧 listen — observability           │
  ├─────────────────────────────────────────────┤
  │  mode          ${config.mode.padEnd(29)}│
  │  audio device  ${audioLine.slice(0, 29).padEnd(29)}│
  │  transcriber   ${transcriber.slice(0, 29).padEnd(29)}│
  │  gate model    ${config.gateModel.padEnd(29)}│
  │  analysis      ${config.analysisModel.padEnd(29)}│
  │  threshold     ${(config.threshold + "/10").padEnd(29)}│
  │  gate          ${(config.localGateEndpoint ? "local → " + config.gateModel : "remote → " + config.gateModel).slice(0, 29).padEnd(29)}│
  │  buffer        ${(config.bufferMinutes + " min").padEnd(29)}│
  │  watchlist     ${(config.watchlistPath || "disabled").padEnd(29)}│
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
  const { watchlist, events, session } = await initSystems(config);

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

      // 3. WATCHLIST — instant string match, no LLM
      await checkWatchlist(
        chunk.text,
        buffer.recentText(1),
        entry.id,
        watchlist,
        events,
        session,
        config,
        cycleLabel
      );

      if (!running) break;

      // 4. GATE — run against rolling buffer context
      const shouldGate =
        config.gateEveryNChunks <= 0 ||
        buffer.chunksSinceLastGate >= config.gateEveryNChunks;

      if (shouldGate) {
        const recentContext = buffer.recentText(1);
        if (config.verbose) {
          process.stdout.write(
            `  ${cycleLabel} 🚦 gate (${buffer.wordCount}w)...`
          );
        }

        const t0 = performance.now();
        const gateResult = await runGate(recentContext, config);
        const gateMs = Math.round(performance.now() - t0);
        buffer.resetGateCounter();

        const escalated = gateResult.score >= config.threshold;

        await events.gateCheck(
          gateResult.score,
          gateResult.reason,
          buffer.wordCount,
          escalated
        );
        session.addGate(gateResult, gateMs, escalated, entry.id);

        if (config.verbose) {
          const icon = escalated ? "🟢" : "⚪";
          console.log(
            ` ${icon} ${gateResult.score}/10 (${gateMs}ms) — ${gateResult.reason}`
          );
        }

        // 5. Escalate → full analysis
        if (escalated) {
          if (config.verbose) {
            console.log(
              `  ${cycleLabel} 🚀 analyzing with ${config.analysisModel}...`
            );
          }

          const fullContext = buffer.fullContextTimestamped();
          const result = await analyze(
            fullContext,
            gateResult.reason,
            config
          );

          await events.analysisComplete(gateResult.reason, result.insights);
          session.addAnalysis(result, entry.id);
          await notify(result);
        }
      }
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
// This process runs the dashboard, watchlist, gate, and analysis pipeline.

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

    // Watchlist — instant string match
    await checkWatchlist(
      post.text,
      buffer.recentText(1),
      entry.id,
      watchlist,
      events,
      session,
      config,
      cycleLabel
    );

    if (!running) return;

    // Gate — run on buffer context
    if (buffer.wordCount >= 5) {
      const t0 = performance.now();
      const gateResult = await runGate(buffer.recentText(2), config);
      const gateMs = Math.round(performance.now() - t0);
      buffer.resetGateCounter();

      const escalated = gateResult.score >= config.threshold;
      await events.gateCheck(
        gateResult.score,
        gateResult.reason,
        buffer.wordCount,
        escalated
      );
      session.addGate(gateResult, gateMs, escalated, entry.id);

      if (config.verbose) {
        const icon = escalated ? "🟢" : "⚪";
        console.log(
          `  ${cycleLabel} 🚦 ${icon} ${gateResult.score}/10 (${gateMs}ms) — ${gateResult.reason}`
        );
      }

      if (escalated) {
        if (config.verbose) {
          console.log(
            `  ${cycleLabel} 🚀 analyzing with ${config.analysisModel}...`
          );
        }
        const result = await analyze(
          buffer.fullContextTimestamped(),
          gateResult.reason,
          config
        );
        await events.analysisComplete(gateResult.reason, result.insights);
        session.addAnalysis(result, entry.id);
        await notify(result);
      }
    }
  };

  // Serialized wrapper: queues transcript processing to avoid concurrent mutations
  const serialOnTranscript = (post: TranscriptPost) => {
    pending = pending
      .then(() => onTranscript(post))
      .catch((err) => console.error("  ⚠ transcript callback error:", err));
  };

  // Init systems with the serialized transcript callback
  const { watchlist, events, session } = await initSystems(config, serialOnTranscript);

  // Mark ready — safe to process transcripts now that all systems are initialized
  ready = true;

  console.log("  🌙 moonshine mode — waiting for transcripts from Moonshine app");
  console.log("  📋 POST transcripts to http://localhost:3838/api/transcript");
  console.log("  🖥  Open the Moonshine menu bar app to start transcribing.\n");

  // Keep the process alive until SIGINT sets running=false
  // (The top-level SIGINT handler already saves the session)
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
  const { watchlist, events, session } = await initSystems(config);

  console.log(
    "  📋 pipe mode — paste transcript, press Enter, Ctrl+D to finish."
  );
  console.log("  Watchlist + gate check on every input.\n");

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

    // Watchlist — instant string match
    await checkWatchlist(
      text,
      buffer.recentText(1),
      entry.id,
      watchlist,
      events,
      session,
      config,
      "     "
    );

    // Gate — run on buffer context after every input
    if (buffer.wordCount >= 5) {
      const t0 = performance.now();
      const gateResult = await runGate(buffer.recentText(2), config);
      const gateMs = Math.round(performance.now() - t0);
      buffer.resetGateCounter();

      const escalated = gateResult.score >= config.threshold;
      await events.gateCheck(
        gateResult.score,
        gateResult.reason,
        buffer.wordCount,
        escalated
      );
      session.addGate(gateResult, gateMs, escalated, entry.id);

      console.log(
        `  ${escalated ? "🟢" : "⚪"} ${gateResult.score}/10 (${gateMs}ms) — ${gateResult.reason}`
      );

      if (escalated) {
        console.log(`  🚀 analyzing...`);
        const result = await analyze(
          buffer.fullContextTimestamped(),
          gateResult.reason,
          config
        );
        await events.analysisComplete(gateResult.reason, result.insights);
        session.addAnalysis(result, entry.id);
        await notify(result);
      }
    }
  }

  // Final gate check on remaining text
  if (buffer.wordCount > 20) {
    console.log(`\n  🚦 final gate check on remaining buffer...`);
    const gateResult = await runGate(buffer.fullContext(), config);
    const escalated = gateResult.score >= config.threshold;

    await events.gateCheck(
      gateResult.score,
      gateResult.reason,
      buffer.wordCount,
      escalated
    );

    const lastEntry = session.getTimeline().at(-1);
    if (lastEntry) session.addGate(gateResult, 0, escalated, lastEntry.id);

    console.log(
      `  ${escalated ? "🟢" : "⚪"} score: ${gateResult.score}/10 — ${gateResult.reason}`
    );

    if (escalated) {
      console.log(`  🚀 analyzing...`);
      const result = await analyze(
        buffer.fullContextTimestamped(),
        gateResult.reason,
        config
      );
      await events.analysisComplete(gateResult.reason, result.insights);
      if (lastEntry) session.addAnalysis(result, lastEntry.id);
      await notify(result);
    } else {
      console.log("  ⚪ nothing actionable detected.");
    }
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
