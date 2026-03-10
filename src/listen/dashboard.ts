/**
 * Dashboard — lightweight web UI for the listen session.
 *
 * Serves a live timeline at http://localhost:3838
 * Uses SSE (Server-Sent Events) for real-time updates.
 * Exposes REST endpoints for corrections and session data.
 */

import type { SessionStore } from "./session";
import { buildDimensionMeta } from "./intent-vector";

const DASHBOARD_PORT = 3838;

// ── Transcript POST payload (from Moonshine Swift app) ────────────
export interface TranscriptPost {
  text: string;
  durationSeconds: number;
  lineId: number;
  source: string;
}

/** Callback for incoming transcript POSTs. */
export type OnTranscriptFn = (transcript: TranscriptPost) => void | Promise<void>;

/**
 * Start the dashboard server. Returns the server instance.
 *
 * @param store    Session store for timeline & SSE
 * @param onTranscript  Optional callback fired when a POST /api/transcript arrives.
 *                      Used by --moonshine mode to feed transcripts into the pipeline.
 */
export function startDashboard(
  store: SessionStore,
  onTranscript?: OnTranscriptFn
) {
  const server = Bun.serve({
    port: DASHBOARD_PORT,
    idleTimeout: 255, // max value — SSE connections stay open
    fetch(req) {
      const url = new URL(req.url);

      // ── Transcript POST (Moonshine → pipeline) ─────────────
      if (url.pathname === "/api/transcript" && req.method === "POST") {
        return handleTranscriptPost(req, store, onTranscript);
      }

      // ── SSE stream ─────────────────────────────────────────
      if (url.pathname === "/events") {
        return new Response(
          new ReadableStream({
            start(controller) {
              const encoder = new TextEncoder();
              const send = (event: string, data: unknown) => {
                controller.enqueue(
                  encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
                );
              };

              // Send current state + dimension metadata so dashboard discovers dimensions at runtime
              send("init", {
                session: store.getSession(),
                stats: store.getStats(),
                dimensionMeta: buildDimensionMeta(),
              });

              // Subscribe to updates
              const unsub = store.subscribe((event, data) => {
                try {
                  send(event, data);
                  send("stats", store.getStats());
                } catch {
                  unsub();
                }
              });

              // Cleanup on close
              req.signal.addEventListener("abort", () => unsub());
            },
          }),
          {
            headers: {
              "Content-Type": "text/event-stream",
              "Cache-Control": "no-cache",
              Connection: "keep-alive",
              "Access-Control-Allow-Origin": "*",
            },
          }
        );
      }

      // ── REST: Get session ──────────────────────────────────
      if (url.pathname === "/api/session" && req.method === "GET") {
        return Response.json({
          session: store.getSession(),
          stats: store.getStats(),
        });
      }

      // ── REST: Get router decisions (observability) ─────────
      if (url.pathname === "/api/decisions" && req.method === "GET") {
        const decisions = store.getDecisions();
        const minInterest = Number(url.searchParams.get("minInterest") || 0);
        const skill = url.searchParams.get("skill");
        const limit = Number(url.searchParams.get("limit") || 200);

        let filtered = decisions;
        if (minInterest > 0) {
          filtered = filtered.filter((d) => d.interest >= minInterest);
        }
        if (skill) {
          filtered = filtered.filter((d) =>
            d.matches.some((m) => m.skill === skill)
          );
        }

        return Response.json({
          total: decisions.length,
          filtered: filtered.length,
          decisions: filtered.slice(-limit),
        });
      }

      // ── REST: Correct transcription ────────────────────────
      if (url.pathname.startsWith("/api/correct/") && req.method === "PATCH") {
        return handleCorrection(req, url, store);
      }

      // ── Dashboard HTML ─────────────────────────────────────
      if (url.pathname === "/" || url.pathname === "/dashboard") {
        return new Response(DASHBOARD_HTML, {
          headers: { "Content-Type": "text/html" },
        });
      }

      // ── CORS preflight ──────────────────────────────────────
      if (req.method === "OPTIONS") {
        return new Response(null, {
          status: 204,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PATCH, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
          },
        });
      }

      return new Response("Not found", { status: 404 });
    },
  });

  console.log(`  🖥  dashboard → http://localhost:${DASHBOARD_PORT}`);
  return server;
}

async function handleCorrection(
  req: Request,
  url: URL,
  store: SessionStore
): Promise<Response> {
  const entryId = url.pathname.split("/").pop();
  if (!entryId) return Response.json({ error: "missing entry id" }, { status: 400 });

  try {
    const body = (await req.json()) as { text: string };
    if (!body.text) return Response.json({ error: "missing text" }, { status: 400 });

    const ok = store.correct(entryId, body.text);
    if (!ok) return Response.json({ error: "entry not found" }, { status: 404 });

    return Response.json({ ok: true, entryId, text: body.text });
  } catch {
    return Response.json({ error: "invalid request" }, { status: 400 });
  }
}

async function handleTranscriptPost(
  req: Request,
  store: SessionStore,
  onTranscript?: OnTranscriptFn
): Promise<Response> {
  try {
    const body = (await req.json()) as TranscriptPost;

    if (!body.text || typeof body.text !== "string") {
      return Response.json({ error: "missing or invalid text" }, { status: 400 });
    }

    // Normalize
    const transcript: TranscriptPost = {
      text: body.text.trim(),
      durationSeconds: Number(body.durationSeconds) || 0,
      lineId: Number(body.lineId) || 0,
      source: body.source || "unknown",
    };

    if (!transcript.text) {
      return Response.json({ error: "empty text after trim" }, { status: 400 });
    }

    // Fire callback (moonshine mode pipeline processing)
    if (onTranscript) {
      // Don't await — let the pipeline run async so the Swift app gets a fast response
      Promise.resolve(onTranscript(transcript)).catch((err) => {
        console.error("  ⚠ transcript callback error:", err);
      });
    }

    return Response.json({ ok: true, text: transcript.text, lineId: transcript.lineId });
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }
}

// ── Dashboard HTML ─────────────────────────────────────────────────

const DASHBOARD_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>listen — dashboard</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #7d8590; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --purple: #bc8cff; --pink: #f778ba; --orange: #d18616;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Berkeley Mono', 'SF Mono', 'Fira Code', monospace;
    background: var(--bg); color: var(--text);
    font-size: 13px; line-height: 1.5;
  }

  /* ── Header ──────────────────────────────────────────────── */
  header {
    position: sticky; top: 0; z-index: 10;
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 10px 20px;
  }
  .header-top {
    display: flex; align-items: center; gap: 16px;
  }
  header h1 { font-size: 14px; font-weight: 600; }
  .stats {
    display: flex; gap: 14px; margin-left: auto; font-size: 11px; color: var(--muted);
    flex-wrap: wrap;
  }
  .stats .stat { display: flex; gap: 4px; align-items: center; }
  .stats .num { color: var(--accent); font-weight: 600; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

  /* ── Tabs ─────────────────────────────────────────────────── */
  .tabs {
    display: flex; gap: 0; margin-top: 8px; border-bottom: 1px solid var(--border);
  }
  .tab {
    padding: 6px 16px; font-size: 12px; cursor: pointer; color: var(--muted);
    border-bottom: 2px solid transparent; transition: all 0.15s;
    background: none; border-top: none; border-left: none; border-right: none;
    font-family: inherit;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-badge {
    font-size: 10px; padding: 0 5px; border-radius: 8px;
    background: var(--border); color: var(--muted); margin-left: 4px;
  }

  /* ── Tab Panels ──────────────────────────────────────────── */
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* ── Timeline ────────────────────────────────────────────── */
  #timeline {
    padding: 16px 20px; display: flex; flex-direction: column; gap: 2px;
    max-width: 960px; margin: 0 auto; padding-bottom: 100px;
  }

  .entry {
    display: grid; grid-template-columns: 60px 1fr auto; gap: 8px;
    padding: 6px 10px; border-radius: 6px;
    transition: background 0.15s;
  }
  .entry:hover { background: var(--surface); }
  .entry.has-event { background: rgba(88, 166, 255, 0.05); }
  .entry.has-router { border-left: 2px solid var(--accent); }
  .entry.has-skill { border-left: 2px solid var(--orange); }
  .entry.has-watchlist { background: rgba(248, 81, 73, 0.08); border-left: 2px solid var(--red); }
  .entry.has-escalation { background: rgba(63, 185, 80, 0.08); border-left: 2px solid var(--green); }
  .entry.is-silence { opacity: 0.3; }

  .time { color: var(--muted); font-size: 11px; padding-top: 2px; }
  .text-col { min-width: 0; }
  .transcript {
    cursor: text; padding: 2px 4px; border-radius: 3px;
    border: 1px solid transparent; transition: border-color 0.15s;
  }
  .transcript:hover { border-color: var(--border); }
  .transcript:focus {
    outline: none; border-color: var(--accent);
    background: rgba(88, 166, 255, 0.1);
  }
  .transcript.corrected { color: var(--yellow); font-style: italic; }
  .transcript.corrected::after { content: ' (corrected)'; font-size: 10px; color: var(--muted); }

  .events { display: flex; flex-direction: column; gap: 2px; margin-top: 2px; }
  .evt {
    font-size: 11px; padding: 2px 6px; border-radius: 3px;
    display: inline-flex; align-items: center; gap: 4px; width: fit-content;
  }
  .evt.gate { color: var(--muted); }
  .evt.escalation { color: var(--green); background: rgba(63, 185, 80, 0.1); }
  .evt.watchlist { color: var(--red); background: rgba(248, 81, 73, 0.1); }
  .evt.analysis { color: var(--purple); background: rgba(188, 140, 255, 0.1); }
  .evt.correction { color: var(--yellow); font-size: 10px; }
  .evt.router-evt { color: var(--accent); background: rgba(88, 166, 255, 0.1); cursor: pointer; }
  .evt.router-evt:hover { background: rgba(88, 166, 255, 0.2); }
  .evt.skill-evt { color: var(--orange); background: rgba(209, 134, 22, 0.1); }

  .badge {
    font-size: 10px; padding: 1px 6px; border-radius: 10px;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); white-space: nowrap;
  }

  .analysis-box {
    margin-top: 4px; padding: 8px 10px; border-radius: 6px;
    background: rgba(188, 140, 255, 0.06); border: 1px solid rgba(188, 140, 255, 0.15);
    font-size: 12px; white-space: pre-wrap;
  }

  .empty { text-align: center; color: var(--muted); padding: 60px 20px; }
  .empty h2 { font-size: 16px; margin-bottom: 8px; }

  /* ── Interest bar (inline) ───────────────────────────────── */
  .interest-bar {
    display: inline-block; width: 60px; height: 6px; border-radius: 3px;
    background: var(--border); vertical-align: middle; margin: 0 4px;
    overflow: hidden;
  }
  .interest-fill {
    height: 100%; border-radius: 3px; transition: width 0.3s;
  }

  /* ── Confidence bar ──────────────────────────────────────── */
  .confidence-bar {
    display: inline-block; width: 40px; height: 4px; border-radius: 2px;
    background: var(--border); vertical-align: middle; margin: 0 3px;
    overflow: hidden;
  }
  .confidence-fill {
    height: 100%; border-radius: 2px; background: var(--accent);
  }

  /* ── Context Inspector (expandable) ──────────────────────── */
  .ctx-inspector {
    display: none; margin-top: 6px; padding: 10px 12px; border-radius: 6px;
    background: rgba(88, 166, 255, 0.04); border: 1px solid rgba(88, 166, 255, 0.12);
    font-size: 11px;
  }
  .ctx-inspector.open { display: block; }
  .ctx-section { margin-bottom: 8px; }
  .ctx-label {
    font-size: 10px; color: var(--accent); text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 2px; font-weight: 600;
  }
  .ctx-content {
    color: var(--muted); white-space: pre-wrap; max-height: 120px;
    overflow-y: auto; padding: 4px 6px; background: var(--bg);
    border-radius: 4px; border: 1px solid var(--border);
  }
  .ctx-skills {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px;
  }
  .ctx-skill-card {
    padding: 4px 8px; border-radius: 4px; font-size: 11px;
    border: 1px solid var(--border); background: var(--surface);
  }
  .ctx-skill-card.executed { border-color: var(--green); }
  .ctx-skill-card.failed { border-color: var(--red); }
  .ctx-skill-card .params { color: var(--muted); font-size: 10px; }

  /* ── Decision Log Panel ──────────────────────────────────── */
  #decisions-panel {
    max-width: 960px; margin: 0 auto; padding: 16px 20px;
  }
  .decision-filters {
    display: flex; gap: 12px; align-items: center; margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .decision-filters label { font-size: 11px; color: var(--muted); }
  .decision-filters select, .decision-filters input {
    font-family: inherit; font-size: 12px; padding: 4px 8px;
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px;
  }

  .decision-card {
    padding: 10px 14px; border-radius: 6px; margin-bottom: 6px;
    border: 1px solid var(--border); background: var(--surface);
    cursor: pointer; transition: border-color 0.15s;
  }
  .decision-card:hover { border-color: var(--accent); }
  .decision-card.expanded { border-color: var(--accent); }
  .decision-header {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }
  .decision-time { color: var(--muted); font-size: 11px; min-width: 60px; }
  .decision-interest { font-weight: 600; min-width: 30px; }
  .decision-reason { color: var(--muted); font-size: 12px; flex: 1; min-width: 0; }
  .decision-skills-inline {
    display: flex; gap: 4px; flex-wrap: wrap;
  }
  .decision-skill-badge {
    font-size: 10px; padding: 1px 6px; border-radius: 8px;
    background: rgba(88, 166, 255, 0.15); color: var(--accent);
  }
  .decision-skill-badge.executed { background: rgba(63, 185, 80, 0.15); color: var(--green); }
  .decision-skill-badge.failed { background: rgba(248, 81, 73, 0.15); color: var(--red); }
  .decision-latency {
    font-size: 10px; color: var(--muted);
  }
  .decision-detail {
    display: none; margin-top: 10px; padding-top: 10px;
    border-top: 1px solid var(--border);
  }
  .decision-card.expanded .decision-detail { display: block; }
  .decision-empty {
    text-align: center; color: var(--muted); padding: 40px 20px; font-size: 12px;
  }

  /* ── Stats Panel ─────────────────────────────────────────── */
  .stats-row {
    display: flex; gap: 10px; flex-wrap: wrap; font-size: 11px;
    color: var(--muted); padding: 4px 0;
  }
  .stats-row .sep { color: var(--border); }

  /* ── Intent Vector Panel ─────────────────────────────────── */
  #intent-panel {
    max-width: 960px; margin: 0 auto; padding: 16px 20px;
  }
  .intent-layout {
    display: grid; grid-template-columns: 280px 1fr; gap: 20px;
    margin-bottom: 20px;
  }
  @media (max-width: 700px) {
    .intent-layout { grid-template-columns: 1fr; }
  }
  .radar-container {
    display: flex; flex-direction: column; align-items: center; gap: 8px;
  }
  .radar-container canvas {
    background: var(--surface); border-radius: 8px;
    border: 1px solid var(--border);
  }
  .dimension-readouts {
    display: flex; flex-direction: column; gap: 6px;
  }
  .dim-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 10px; border-radius: 6px;
    background: var(--surface); border: 1px solid var(--border);
  }
  .dim-label {
    font-size: 11px; color: var(--muted); width: 80px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .dim-bar {
    flex: 1; height: 8px; border-radius: 4px;
    background: var(--border); overflow: hidden;
  }
  .dim-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.4s ease-out;
  }
  .dim-value {
    font-size: 12px; font-weight: 600; width: 36px; text-align: right;
  }
  .dim-trend {
    font-size: 12px; width: 16px; text-align: center;
  }
  .sparkline-section {
    margin-top: 16px;
  }
  .sparkline-section h3 {
    font-size: 12px; color: var(--muted); margin-bottom: 8px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .sparkline-row {
    display: flex; align-items: center; gap: 8px; margin-bottom: 4px;
  }
  .sparkline-label {
    font-size: 10px; color: var(--muted); width: 70px;
    text-transform: uppercase;
  }
  .sparkline-canvas {
    background: var(--surface); border-radius: 4px;
    border: 1px solid var(--border);
  }
  .intent-empty {
    text-align: center; color: var(--muted); padding: 60px 20px;
  }
  .intent-empty h2 { font-size: 16px; margin-bottom: 8px; }

  /* ── Gate Status ─────────────────────────────────────────── */
  .gate-status {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 6px; margin-bottom: 16px;
    border: 1px solid var(--border); background: var(--surface);
    font-size: 12px;
  }
  .gate-indicator {
    width: 10px; height: 10px; border-radius: 50%;
    flex-shrink: 0;
  }
  .gate-indicator.idle { background: var(--muted); }
  .gate-indicator.vigilant { background: var(--yellow); animation: pulse 1.5s infinite; }
  .gate-indicator.active { background: var(--red); animation: pulse 0.8s infinite; }
  .gate-label { font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .gate-label.idle { color: var(--muted); }
  .gate-label.vigilant { color: var(--yellow); }
  .gate-label.active { color: var(--red); }
  .gate-detail { color: var(--muted); font-size: 11px; }
  .gate-promoted {
    padding: 2px 8px; border-radius: 10px; font-size: 10px;
    background: rgba(248, 81, 73, 0.15); color: var(--red);
    font-weight: 600;
  }
</style>
</head>
<body>

<header>
  <div class="header-top">
    <div class="dot"></div>
    <h1>listen</h1>
    <div class="stats">
      <div class="stat">chunks <span class="num" id="s-chunks">0</span></div>
      <div class="stat">words <span class="num" id="s-words">0</span></div>
      <div class="stat">decisions <span class="num" id="s-decisions">0</span></div>
      <div class="stat">skills <span class="num" id="s-skills">0</span></div>
      <div class="stat" title="avg router latency">latency <span class="num" id="s-latency">0</span>ms</div>
      <div class="stat">interest <span class="num" id="s-interest">0</span></div>
      <div class="stat">watchlist <span class="num" id="s-watchlist">0</span></div>
      <div class="stat">escalations <span class="num" id="s-escalations">0</span></div>
    </div>
  </div>
  <div class="tabs">
    <button class="tab active" data-tab="timeline">Timeline</button>
    <button class="tab" data-tab="decisions">Decisions <span class="tab-badge" id="decisions-count">0</span></button>
    <button class="tab" data-tab="intent">Intent</button>
  </div>
</header>

<!-- Timeline Panel -->
<div class="tab-panel active" id="panel-timeline">
  <div id="timeline">
    <div class="empty" id="empty-state">
      <h2>listening...</h2>
      <p>Transcriptions will appear here in real time.</p>
    </div>
  </div>
</div>

<!-- Decisions Panel -->
<div class="tab-panel" id="panel-decisions">
  <div id="decisions-panel">
    <div class="decision-filters">
      <label>min interest:</label>
      <input type="range" id="filter-interest" min="0" max="10" value="0" style="width:80px">
      <span id="filter-interest-val" style="font-size:11px;color:var(--accent);width:20px">0</span>
      <label>skill:</label>
      <select id="filter-skill"><option value="">all</option></select>
      <label>show:</label>
      <select id="filter-show">
        <option value="all">all</option>
        <option value="with-skills">with skill matches</option>
        <option value="escalated">escalated only</option>
      </select>
    </div>
    <div id="decisions-list">
      <div class="decision-empty">No router decisions yet. Speak and they'll appear here.</div>
    </div>
  </div>
</div>

<!-- Intent Vector Panel -->
<div class="tab-panel" id="panel-intent">
  <div id="intent-panel">
    <div class="gate-status" id="gate-status">
      <div class="gate-indicator idle" id="gate-indicator"></div>
      <span class="gate-label idle" id="gate-label">idle</span>
      <span class="gate-detail" id="gate-detail">Activation gate inactive</span>
    </div>
    <div class="intent-layout">
      <div class="radar-container">
        <canvas id="radar-chart" width="260" height="260"></canvas>
      </div>
      <div>
        <div class="dimension-readouts" id="dim-readouts"></div>
        <div class="sparkline-section">
          <h3>Traces (last 5 min)</h3>
          <div id="sparklines"></div>
        </div>
      </div>
    </div>
    <div class="intent-empty" id="intent-empty">
      <h2>no intent data yet</h2>
      <p>Intent vector will appear here as transcripts are processed.</p>
    </div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────
const timeline = document.getElementById('timeline');
const emptyState = document.getElementById('empty-state');
const decisionsList = document.getElementById('decisions-list');
let autoScroll = true;
const allDecisions = [];
const knownSkills = new Set();

// ── Tab switching ──────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
  });
});

// ── Scroll ─────────────────────────────────────────────────────
window.addEventListener('scroll', () => {
  const atBottom = (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 100);
  autoScroll = atBottom;
});

function scrollBottom() {
  if (autoScroll) window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

// ── Stats ──────────────────────────────────────────────────────
function updateStats(stats) {
  document.getElementById('s-chunks').textContent = stats.transcribedChunks || 0;
  document.getElementById('s-words').textContent = stats.totalWords || 0;
  document.getElementById('s-watchlist').textContent = stats.watchlistHits || 0;
  document.getElementById('s-escalations').textContent = stats.escalations || 0;
  document.getElementById('s-decisions').textContent = stats.totalDecisions || 0;
  document.getElementById('s-skills').textContent = stats.skillActivations || 0;
  document.getElementById('s-latency').textContent = stats.avgLatencyMs || 0;
  document.getElementById('s-interest').textContent = stats.avgInterest || 0;
  document.getElementById('decisions-count').textContent = stats.totalDecisions || 0;
}

// ── Helpers ────────────────────────────────────────────────────
function formatTime(iso) {
  return new Date(iso).toLocaleTimeString('en-US', { hour12: false, hour:'2-digit', minute:'2-digit', second:'2-digit' });
}

function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
          .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function interestColor(n) {
  if (n >= 7) return 'var(--green)';
  if (n >= 4) return 'var(--yellow)';
  if (n >= 1) return 'var(--muted)';
  return 'var(--border)';
}

function interestBarHtml(n) {
  const pct = Math.min(100, n * 10);
  return '<span class="interest-bar"><span class="interest-fill" style="width:' + pct + '%;background:' + interestColor(n) + '"></span></span>';
}

function confidenceBarHtml(n) {
  const pct = Math.min(100, n * 100);
  return '<span class="confidence-bar"><span class="confidence-fill" style="width:' + pct + '%"></span></span>';
}

// ── Timeline: Render Entry ─────────────────────────────────────
function renderEntry(entry) {
  if (emptyState) emptyState.remove();

  const existing = document.getElementById(entry.id);
  if (existing) { updateEntry(existing, entry); return; }

  const div = document.createElement('div');
  div.id = entry.id;
  div.className = 'entry' + classesForEntry(entry);

  const isSilence = !entry.original;

  div.innerHTML =
    '<div class="time">' + formatTime(entry.timestamp) + '</div>' +
    '<div class="text-col">' +
      (isSilence
        ? '<span style="color:var(--muted);font-style:italic">(silence)</span>'
        : '<span class="transcript' + (entry.corrected ? ' corrected' : '') + '" contenteditable="true" data-id="' + entry.id + '">' + escHtml(entry.text) + '</span>') +
      '<div class="events" id="events-' + entry.id + '">' + renderEvents(entry.events) + '</div>' +
    '</div>' +
    '<div>' + (entry.durationSeconds ? '<span class="badge">' + entry.durationSeconds + 's</span>' : '') + '</div>';

  timeline.appendChild(div);

  // Bind correction on blur
  const span = div.querySelector('.transcript');
  if (span) {
    span.addEventListener('blur', async (e) => {
      const newText = e.target.textContent.trim();
      const id = e.target.dataset.id;
      if (newText && newText !== entry.original) {
        await fetch('/api/correct/' + id, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: newText }),
        });
        e.target.classList.add('corrected');
      }
    });
    span.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); e.target.blur(); }
      if (e.key === 'Escape') { e.target.textContent = entry.original; e.target.blur(); }
    });
  }

  scrollBottom();
}

function updateEntry(el, data) {
  el.className = 'entry' + classesForEntry(data);
}

function classesForEntry(entry) {
  const evts = entry.events || [];
  if (!entry.original && !entry.text) return ' is-silence';
  if (evts.some(e => e.type === 'watchlist')) return ' has-watchlist';
  if (evts.some(e => e.type === 'gate.escalation')) return ' has-escalation';
  if (evts.some(e => e.type === 'skill')) return ' has-skill';
  if (evts.some(e => e.type === 'router')) return ' has-router';
  if (evts.some(e => e.type !== 'silence' && e.type !== 'gate')) return ' has-event';
  return '';
}

// ── Timeline: Render Events ────────────────────────────────────
function renderEvents(events) {
  return events.map(e => {
    switch (e.type) {
      case 'gate':
        return '<div class="evt gate">score ' + e.score + '/10 (' + e.latencyMs + 'ms) -- ' + escHtml(e.reason) + '</div>';
      case 'gate.escalation':
        return '<div class="evt escalation">ESCALATED ' + e.score + '/10 (' + e.latencyMs + 'ms) -- ' + escHtml(e.reason) + '</div>';
      case 'router':
        return '<div class="evt router-evt" data-entry="' + (e.entryId || '') + '">'
          + interestBarHtml(e.interest) + ' ' + e.interest + '/10'
          + ' (' + e.latencyMs + 'ms) -- ' + escHtml(e.reason)
          + (e.skills && e.skills.length ? ' [' + e.skills.join(', ') + ']' : '')
          + '</div>';
      case 'skill':
        return '<div class="evt skill-evt">'
          + (e.success ? 'OK' : 'FAIL') + ' ' + e.skill + '.' + e.action
          + (e.voice ? ' -- "' + escHtml(e.voice) + '"' : '')
          + '</div>';
      case 'watchlist':
        return '<div class="evt watchlist">[' + e.severity + '] ' + e.category + '/' + e.patternId + ' -- "' + escHtml(e.trigger) + '"</div>';
      case 'analysis':
        return '<div class="evt analysis">analysis</div><div class="analysis-box">' + escHtml(e.insights) + '</div>';
      case 'correction':
        return '<div class="evt correction">corrected: "' + escHtml(e.from).slice(0,40) + '..." -- "' + escHtml(e.to).slice(0,40) + '..."</div>';
      case 'silence': return '';
      default: return '';
    }
  }).join('');
}

function addEventToEntry(entryId, evt) {
  const eventsDiv = document.getElementById('events-' + entryId);
  if (eventsDiv) {
    eventsDiv.innerHTML += renderEvents([evt]);
    const parent = eventsDiv.closest('.entry');
    if (parent) {
      if (evt.type === 'watchlist') parent.className = 'entry has-watchlist';
      else if (evt.type === 'gate.escalation') parent.className = 'entry has-escalation';
      else if (evt.type === 'skill') parent.className = 'entry has-skill';
      else if (evt.type === 'router') parent.className = 'entry has-router';
    }
  }
  scrollBottom();
}

// ── Decision Log ───────────────────────────────────────────────
function addDecision(d) {
  allDecisions.push(d);
  d.matches.forEach(m => {
    if (!knownSkills.has(m.skill)) {
      knownSkills.add(m.skill);
      const opt = document.createElement('option');
      opt.value = m.skill;
      opt.textContent = m.skill;
      document.getElementById('filter-skill').appendChild(opt);
    }
  });
  renderDecisionsList();
}

function updateDecision(update) {
  const d = allDecisions.find(d => d.id === update.decisionId);
  if (!d) return;
  const m = d.matches.find(m => m.skill === update.skill && m.action === update.action);
  if (m) {
    m.executed = update.executed;
    m.success = update.success;
    m.voice = update.voice;
  }
  // Re-render this card if it exists
  const card = document.getElementById(d.id);
  if (card) {
    const badges = card.querySelector('.decision-skills-inline');
    if (badges) badges.innerHTML = renderDecisionSkillBadges(d.matches);
  }
}

function getFilteredDecisions() {
  const minInterest = Number(document.getElementById('filter-interest').value);
  const skill = document.getElementById('filter-skill').value;
  const show = document.getElementById('filter-show').value;

  return allDecisions.filter(d => {
    if (d.interest < minInterest) return false;
    if (skill && !d.matches.some(m => m.skill === skill)) return false;
    if (show === 'with-skills' && d.matches.length === 0) return false;
    if (show === 'escalated' && !d.escalated) return false;
    return true;
  });
}

function renderDecisionSkillBadges(matches) {
  if (!matches.length) return '<span style="color:var(--muted);font-size:10px">no skills matched</span>';
  return matches.map(m => {
    let cls = 'decision-skill-badge';
    if (m.executed && m.success) cls += ' executed';
    if (m.executed && !m.success) cls += ' failed';
    return '<span class="' + cls + '">' + m.skill + '.' + m.action
      + ' ' + (m.confidence * 100).toFixed(0) + '%'
      + (m.executed ? (m.success ? ' OK' : ' FAIL') : '')
      + '</span>';
  }).join('');
}

function renderDecisionCard(d) {
  const card = document.createElement('div');
  card.id = d.id;
  card.className = 'decision-card';

  const iColor = interestColor(d.interest);

  card.innerHTML =
    '<div class="decision-header">' +
      '<span class="decision-time">' + formatTime(d.timestamp) + '</span>' +
      interestBarHtml(d.interest) +
      '<span class="decision-interest" style="color:' + iColor + '">' + d.interest + '/10</span>' +
      '<span class="decision-reason">' + escHtml(d.reason) + '</span>' +
      '<span class="decision-skills-inline">' + renderDecisionSkillBadges(d.matches) + '</span>' +
      '<span class="decision-latency">' + d.latencyMs + 'ms</span>' +
      (d.escalated ? '<span class="badge" style="border-color:var(--green);color:var(--green)">escalated</span>' : '') +
    '</div>' +
    '<div class="decision-detail">' +
      '<div class="ctx-section">' +
        '<div class="ctx-label">transcript</div>' +
        '<div class="ctx-content">' + escHtml(d.transcript) + '</div>' +
      '</div>' +
      (d.bufferContext ? (
        '<div class="ctx-section">' +
          '<div class="ctx-label">buffer context (recent conversation)</div>' +
          '<div class="ctx-content">' + escHtml(d.bufferContext) + '</div>' +
        '</div>'
      ) : '') +
      (d.skillState ? (
        '<div class="ctx-section">' +
          '<div class="ctx-label">skill state</div>' +
          '<div class="ctx-content">' + escHtml(d.skillState) + '</div>' +
        '</div>'
      ) : '') +
      (d.recentSkills && d.recentSkills.length ? (
        '<div class="ctx-section">' +
          '<div class="ctx-label">recent skill history</div>' +
          '<div class="ctx-content">' + d.recentSkills.map(s =>
            s.skill + '.' + s.action + ' ' + (s.success ? 'OK' : 'FAIL') + ' (' + s.agoSeconds + 's ago)' + (s.voice ? ' -- "' + escHtml(s.voice) + '"' : '')
          ).join('\\n') + '</div>' +
        '</div>'
      ) : '') +
      '<div class="ctx-section">' +
        '<div class="ctx-label">skill matches (' + d.matches.length + ')</div>' +
        '<div class="ctx-skills">' + d.matches.map(m => {
          let cls = 'ctx-skill-card';
          if (m.executed && m.success) cls += ' executed';
          if (m.executed && !m.success) cls += ' failed';
          const paramStr = Object.keys(m.params || {}).length
            ? '<div class="params">' + Object.entries(m.params).map(([k,v]) => k + '=' + escHtml(v)).join(', ') + '</div>'
            : '';
          return '<div class="' + cls + '">'
            + '<strong>' + m.skill + '.' + m.action + '</strong> '
            + confidenceBarHtml(m.confidence) + ' ' + (m.confidence * 100).toFixed(0) + '%'
            + (m.executed != null ? '<br>' + (m.success ? '<span style="color:var(--green)">executed OK</span>' : '<span style="color:var(--red)">failed</span>') : '')
            + (m.voice ? '<br><span style="color:var(--muted)">voice: "' + escHtml(m.voice) + '"</span>' : '')
            + paramStr
            + '</div>';
        }).join('') + '</div>' +
      '</div>' +
      '<div class="stats-row">' +
        '<span>words: ' + d.wordCount + '</span>' +
        '<span class="sep">|</span>' +
        '<span>latency: ' + d.latencyMs + 'ms</span>' +
        '<span class="sep">|</span>' +
        '<span>escalated: ' + (d.escalated ? 'yes' : 'no') + '</span>' +
        '<span class="sep">|</span>' +
        '<span>entry: ' + d.entryId + '</span>' +
      '</div>' +
    '</div>';

  card.addEventListener('click', () => card.classList.toggle('expanded'));
  return card;
}

function renderDecisionsList() {
  const filtered = getFilteredDecisions();
  decisionsList.innerHTML = '';

  if (filtered.length === 0) {
    decisionsList.innerHTML = '<div class="decision-empty">No decisions match filters.' +
      (allDecisions.length > 0 ? ' (' + allDecisions.length + ' total)' : '') + '</div>';
    return;
  }

  // Render newest first
  for (let i = filtered.length - 1; i >= 0; i--) {
    decisionsList.appendChild(renderDecisionCard(filtered[i]));
  }
}

// Filters
document.getElementById('filter-interest').addEventListener('input', (e) => {
  document.getElementById('filter-interest-val').textContent = e.target.value;
  renderDecisionsList();
});
document.getElementById('filter-skill').addEventListener('change', renderDecisionsList);
document.getElementById('filter-show').addEventListener('change', renderDecisionsList);

// ── Intent Vector ─────────────────────────────────────────────
// Dimension metadata is received from the server via SSE init event.
// These are populated dynamically — no hardcoded dimension list.
let DIMS = [];        // string[] of dimension keys
let DIM_COLORS = {};  // key → hex color
let DIM_LABELS = {};  // key → short label
let DIM_RANGES = {};  // key → { min, max }

function setDimensionMeta(meta) {
  if (!meta || !meta.length) return;
  DIMS = meta.map(d => d.key);
  DIM_COLORS = {};
  DIM_LABELS = {};
  DIM_RANGES = {};
  meta.forEach(d => {
    DIM_COLORS[d.key] = d.color;
    DIM_LABELS[d.key] = d.shortLabel;
    DIM_RANGES[d.key] = { min: d.min ?? 0, max: d.max ?? 1 };
  });
}

let intentHistory = [];
let currentIntent = null;

// -- Radar chart -------------------------------------------------------
const radarCanvas = document.getElementById('radar-chart');
const radarCtx = radarCanvas.getContext('2d');

// Normalize a dimension value to [0, 1] for radar display
function normalizeDim(key, val) {
  const range = DIM_RANGES[key] || { min: 0, max: 1 };
  if (range.max === range.min) return 0;
  return (val - range.min) / (range.max - range.min);
}

function drawRadar(dims) {
  const W = radarCanvas.width;
  const H = radarCanvas.height;
  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(cx, cy) - 30;
  const n = DIMS.length;

  radarCtx.clearRect(0, 0, W, H);
  if (n === 0) return; // no dimensions registered yet

  // Grid rings (0.25, 0.5, 0.75, 1.0)
  radarCtx.strokeStyle = '#30363d';
  radarCtx.lineWidth = 1;
  for (let ring = 0.25; ring <= 1; ring += 0.25) {
    radarCtx.beginPath();
    for (let i = 0; i <= n; i++) {
      const angle = (Math.PI * 2 * (i % n)) / n - Math.PI / 2;
      const x = cx + R * ring * Math.cos(angle);
      const y = cy + R * ring * Math.sin(angle);
      if (i === 0) radarCtx.moveTo(x, y);
      else radarCtx.lineTo(x, y);
    }
    radarCtx.stroke();
  }

  // Axis lines + labels
  radarCtx.font = '11px "Berkeley Mono", "SF Mono", monospace';
  radarCtx.fillStyle = '#7d8590';
  for (let i = 0; i < n; i++) {
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    const lx = cx + R * Math.cos(angle);
    const ly = cy + R * Math.sin(angle);
    radarCtx.beginPath();
    radarCtx.moveTo(cx, cy);
    radarCtx.lineTo(lx, ly);
    radarCtx.stroke();

    // Label
    const labelR = R + 16;
    const tx = cx + labelR * Math.cos(angle);
    const ty = cy + labelR * Math.sin(angle);
    radarCtx.textAlign = Math.abs(Math.cos(angle)) < 0.01 ? 'center' : (Math.cos(angle) > 0 ? 'left' : 'right');
    radarCtx.textBaseline = Math.abs(Math.sin(angle)) < 0.01 ? 'middle' : (Math.sin(angle) > 0 ? 'top' : 'bottom');
    radarCtx.fillText(DIM_LABELS[DIMS[i]] || DIMS[i], tx, ty);
  }

  if (!dims) return;

  // Filled polygon — normalize all values to [0, 1] for display
  radarCtx.beginPath();
  for (let i = 0; i <= n; i++) {
    const key = DIMS[i % n];
    const angle = (Math.PI * 2 * (i % n)) / n - Math.PI / 2;
    const val = normalizeDim(key, dims[key] || 0);
    const x = cx + R * val * Math.cos(angle);
    const y = cy + R * val * Math.sin(angle);
    if (i === 0) radarCtx.moveTo(x, y);
    else radarCtx.lineTo(x, y);
  }
  radarCtx.fillStyle = 'rgba(88, 166, 255, 0.15)';
  radarCtx.fill();
  radarCtx.strokeStyle = '#58a6ff';
  radarCtx.lineWidth = 2;
  radarCtx.stroke();

  // Points
  for (let i = 0; i < n; i++) {
    const key = DIMS[i];
    const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
    const val = normalizeDim(key, dims[key] || 0);
    const x = cx + R * val * Math.cos(angle);
    const y = cy + R * val * Math.sin(angle);
    radarCtx.beginPath();
    radarCtx.arc(x, y, 4, 0, Math.PI * 2);
    radarCtx.fillStyle = DIM_COLORS[key] || '#58a6ff';
    radarCtx.fill();
    radarCtx.strokeStyle = '#0d1117';
    radarCtx.lineWidth = 1.5;
    radarCtx.stroke();
  }
}

// -- Dimension readouts ------------------------------------------------
const readoutsContainer = document.getElementById('dim-readouts');
function renderReadouts(dims, trends) {
  readoutsContainer.innerHTML = '';
  DIMS.forEach(key => {
    const val = dims ? (dims[key] || 0) : 0;
    const trend = trends ? (trends[key] || 0) : 0;
    const trendArrow = trend > 0.05 ? '↑' : (trend < -0.05 ? '↓' : '→');
    const trendColor = trend > 0.05 ? 'var(--green)' : (trend < -0.05 ? 'var(--red)' : 'var(--muted)');
    // Normalize to percentage for the bar display
    const pct = Math.min(100, normalizeDim(key, val) * 100);
    const color = DIM_COLORS[key] || '#58a6ff';

    const row = document.createElement('div');
    row.className = 'dim-row';
    row.innerHTML =
      '<span class="dim-label">' + (DIM_LABELS[key] || key) + '</span>' +
      '<div class="dim-bar"><div class="dim-fill" style="width:' + pct + '%;background:' + color + '"></div></div>' +
      '<span class="dim-value" style="color:' + color + '">' + val.toFixed(2) + '</span>' +
      '<span class="dim-trend" style="color:' + trendColor + '">' + trendArrow + '</span>';
    readoutsContainer.appendChild(row);
  });
}

// -- Sparklines --------------------------------------------------------
const sparklinesContainer = document.getElementById('sparklines');

function renderSparklines(history) {
  sparklinesContainer.innerHTML = '';
  if (!history || history.length < 2 || DIMS.length === 0) return;

  const W = 500;
  const H = 28;

  DIMS.forEach(key => {
    const row = document.createElement('div');
    row.className = 'sparkline-row';

    const label = document.createElement('span');
    label.className = 'sparkline-label';
    label.textContent = DIM_LABELS[key] || key;
    row.appendChild(label);

    const canvas = document.createElement('canvas');
    canvas.className = 'sparkline-canvas';
    canvas.width = W;
    canvas.height = H;
    row.appendChild(canvas);
    sparklinesContainer.appendChild(row);

    const ctx = canvas.getContext('2d');
    // Normalize values to [0, 1] for sparkline display
    const values = history.map(s => normalizeDim(key, s.dimensions[key] || 0));

    // Fill area
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let i = 0; i < values.length; i++) {
      const x = (i / (values.length - 1)) * W;
      const y = H - values[i] * (H - 2);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(W, H);
    ctx.closePath();
    // Hex to rgba fill
    const hex = DIM_COLORS[key] || '#58a6ff';
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.12)';
    ctx.fill();

    // Line
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = (i / (values.length - 1)) * W;
      const y = H - values[i] * (H - 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = hex;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

// -- Gate status -------------------------------------------------------
function updateGateStatus(gate) {
  const indicator = document.getElementById('gate-indicator');
  const label = document.getElementById('gate-label');
  const detail = document.getElementById('gate-detail');
  const statusEl = document.getElementById('gate-status');

  if (!gate) {
    indicator.className = 'gate-indicator idle';
    label.className = 'gate-label idle';
    label.textContent = 'idle';
    detail.textContent = 'Activation gate inactive';
    // Remove any promoted badge
    const old = statusEl.querySelector('.gate-promoted');
    if (old) old.remove();
    return;
  }

  // Update indicator + label
  indicator.className = 'gate-indicator ' + gate.state;
  label.className = 'gate-label ' + gate.state;
  label.textContent = gate.state;

  // Build detail text
  const dimName = gate.targetDimension;
  const level = gate.activationLevel;
  let detailText = dimName + ': ' + level.toFixed(2) +
    ' | threshold: ' + gate.effectiveThreshold.toFixed(2);
  if (gate.timeSinceLastTrigger !== null) {
    const ago = Math.round(gate.timeSinceLastTrigger / 1000);
    detailText += ' | last trigger: ' + ago + 's ago';
  }
  detail.textContent = detailText;

  // Promoted badge
  const oldBadge = statusEl.querySelector('.gate-promoted');
  if (oldBadge) oldBadge.remove();
  if (gate.promoted) {
    const badge = document.createElement('span');
    badge.className = 'gate-promoted';
    badge.textContent = 'PROMOTED @ ' + (gate.promotedConfidence || 0).toFixed(2);
    statusEl.appendChild(badge);
  }
}

// -- Update intent display ----------------------------------------------
function updateIntent(snapshot, history, gate) {
  const emptyEl = document.getElementById('intent-empty');
  if (snapshot) {
    if (emptyEl) emptyEl.style.display = 'none';
    currentIntent = snapshot;
    intentHistory = history || intentHistory;
    drawRadar(snapshot.dimensions);
    renderReadouts(snapshot.dimensions, snapshot.trends);
    renderSparklines(intentHistory);
    updateGateStatus(gate || null);
  }
}

// Draw initial empty radar
drawRadar(null);
renderReadouts(null, null);
updateGateStatus(null);

// ── SSE Connection ─────────────────────────────────────────────
const evtSource = new EventSource('/events');

evtSource.addEventListener('init', (e) => {
  const { session, stats, dimensionMeta } = JSON.parse(e.data);

  // Apply dimension metadata from server — drives all chart rendering
  if (dimensionMeta) {
    setDimensionMeta(dimensionMeta);
  }

  // Clear all state on (re)init — prevents duplicates on SSE reconnect
  timeline.innerHTML = '';
  allDecisions.length = 0;
  knownSkills.clear();
  document.getElementById('filter-skill').innerHTML = '<option value="">all</option>';

  updateStats(stats);
  session.timeline.forEach(renderEntry);

  // Load existing decisions
  if (session.decisions) {
    session.decisions.forEach(d => {
      allDecisions.push(d);
      d.matches.forEach(m => knownSkills.add(m.skill));
    });
    // Rebuild skill filter
    knownSkills.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      document.getElementById('filter-skill').appendChild(opt);
    });
    renderDecisionsList();
  }

  // Load intent vector state
  if (session.intentVector) {
    updateIntent(session.intentVector, session.intentVectorHistory || [], null);
  }

  // Re-draw radar with current dimension set
  drawRadar(null);
  renderReadouts(null, null);
});

evtSource.addEventListener('chunk', (e) => {
  renderEntry(JSON.parse(e.data));
});

evtSource.addEventListener('gate', (e) => {
  const data = JSON.parse(e.data);
  addEventToEntry(data.entryId, data);
});

evtSource.addEventListener('router', (e) => {
  const data = JSON.parse(e.data);
  addEventToEntry(data.entryId, data);
});

evtSource.addEventListener('skill', (e) => {
  const data = JSON.parse(e.data);
  addEventToEntry(data.entryId, data);
});

evtSource.addEventListener('decision', (e) => {
  addDecision(JSON.parse(e.data));
});

evtSource.addEventListener('decision_update', (e) => {
  updateDecision(JSON.parse(e.data));
});

evtSource.addEventListener('watchlist', (e) => {
  const data = JSON.parse(e.data);
  addEventToEntry(data.entryId, data);
});

evtSource.addEventListener('analysis', (e) => {
  const data = JSON.parse(e.data);
  addEventToEntry(data.entryId, data);
});

evtSource.addEventListener('correction', (e) => {
  const data = JSON.parse(e.data);
  addEventToEntry(data.entryId, data);
});

evtSource.addEventListener('intentVector', (e) => {
  const data = JSON.parse(e.data);
  updateIntent(data.snapshot, data.history, data.gate);
});

evtSource.addEventListener('stats', (e) => {
  updateStats(JSON.parse(e.data));
});

evtSource.onerror = () => {
  console.log('SSE reconnecting...');
};
</script>
</body>
</html>`;
