/**
 * Dashboard — lightweight web UI for the listen session.
 *
 * Serves a live timeline at http://localhost:3838
 * Uses SSE (Server-Sent Events) for real-time updates.
 * Exposes REST endpoints for corrections and session data.
 */

import type { SessionStore } from "./session";

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

              // Send current state
              send("init", {
                session: store.getSession(),
                stats: store.getStats(),
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
    --purple: #bc8cff; --pink: #f778ba;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Berkeley Mono', 'SF Mono', 'Fira Code', monospace;
    background: var(--bg); color: var(--text);
    font-size: 13px; line-height: 1.5;
  }
  header {
    position: sticky; top: 0; z-index: 10;
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 12px 20px; display: flex; align-items: center; gap: 16px;
  }
  header h1 { font-size: 14px; font-weight: 600; }
  .stats {
    display: flex; gap: 16px; margin-left: auto; font-size: 11px; color: var(--muted);
  }
  .stats .stat { display: flex; gap: 4px; align-items: center; }
  .stats .num { color: var(--accent); font-weight: 600; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

  #timeline {
    padding: 16px 20px; display: flex; flex-direction: column; gap: 2px;
    max-width: 900px; margin: 0 auto;
  }

  .entry {
    display: grid; grid-template-columns: 60px 1fr auto; gap: 8px;
    padding: 6px 10px; border-radius: 6px;
    transition: background 0.15s;
  }
  .entry:hover { background: var(--surface); }
  .entry.has-event { background: rgba(88, 166, 255, 0.05); }
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

  /* Scroll to bottom behavior */
  #timeline { padding-bottom: 100px; }
</style>
</head>
<body>

<header>
  <div class="dot"></div>
  <h1>🎧 listen</h1>
  <div class="stats">
    <div class="stat">chunks <span class="num" id="s-chunks">0</span></div>
    <div class="stat">words <span class="num" id="s-words">0</span></div>
    <div class="stat">🫀 <span class="num" id="s-watchlist">0</span></div>
    <div class="stat">🚀 <span class="num" id="s-escalations">0</span></div>
    <div class="stat">✏️ <span class="num" id="s-corrections">0</span></div>
  </div>
</header>

<div id="timeline">
  <div class="empty" id="empty-state">
    <h2>listening...</h2>
    <p>Transcriptions will appear here in real time.</p>
  </div>
</div>

<script>
const timeline = document.getElementById('timeline');
const emptyState = document.getElementById('empty-state');
let autoScroll = true;

// Track scroll intent
window.addEventListener('scroll', () => {
  const atBottom = (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 100);
  autoScroll = atBottom;
});

function scrollBottom() {
  if (autoScroll) window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function updateStats(stats) {
  document.getElementById('s-chunks').textContent = stats.transcribedChunks;
  document.getElementById('s-words').textContent = stats.totalWords;
  document.getElementById('s-watchlist').textContent = stats.watchlistHits;
  document.getElementById('s-escalations').textContent = stats.escalations;
  document.getElementById('s-corrections').textContent = stats.corrections;
}

function formatTime(iso) {
  return new Date(iso).toLocaleTimeString('en-US', { hour12: false, hour:'2-digit', minute:'2-digit', second:'2-digit' });
}

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
  const eventsDiv = document.getElementById('events-' + data.entryId || data.id);
  // Just re-render events if we got an update for existing entry
}

function classesForEntry(entry) {
  const evts = entry.events || [];
  if (!entry.original && !entry.text) return ' is-silence';
  if (evts.some(e => e.type === 'watchlist')) return ' has-watchlist';
  if (evts.some(e => e.type === 'gate.escalation')) return ' has-escalation';
  if (evts.some(e => e.type !== 'silence' && e.type !== 'gate')) return ' has-event';
  return '';
}

function renderEvents(events) {
  return events.map(e => {
    switch (e.type) {
      case 'gate':
        return '<div class="evt gate">🚦 ' + e.score + '/10 (' + e.latencyMs + 'ms) — ' + escHtml(e.reason) + '</div>';
      case 'gate.escalation':
        return '<div class="evt escalation">🟢 ' + e.score + '/10 (' + e.latencyMs + 'ms) — ' + escHtml(e.reason) + '</div>';
      case 'watchlist':
        return '<div class="evt watchlist">🫀 [' + e.severity + '] ' + e.category + '/' + e.patternId + ' → "' + escHtml(e.trigger) + '"</div>';
      case 'analysis':
        return '<div class="evt analysis">📊 analysis</div><div class="analysis-box">' + escHtml(e.insights) + '</div>';
      case 'correction':
        return '<div class="evt correction">✏️ corrected: "' + escHtml(e.from).slice(0,40) + '…" → "' + escHtml(e.to).slice(0,40) + '…"</div>';
      case 'silence': return '';
      default: return '';
    }
  }).join('');
}

function addEventToEntry(entryId, evt) {
  const eventsDiv = document.getElementById('events-' + entryId);
  if (eventsDiv) {
    eventsDiv.innerHTML += renderEvents([evt]);
    // Update parent classes
    const parent = eventsDiv.closest('.entry');
    if (parent && evt.type === 'watchlist') parent.className = 'entry has-watchlist';
    if (parent && evt.type === 'gate.escalation') parent.className = 'entry has-escalation';
  }
  scrollBottom();
}

function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── SSE Connection ─────────────────────────────────────────────
const evtSource = new EventSource('/events');

evtSource.addEventListener('init', (e) => {
  const { session, stats } = JSON.parse(e.data);
  updateStats(stats);
  session.timeline.forEach(renderEntry);
});

evtSource.addEventListener('chunk', (e) => {
  renderEntry(JSON.parse(e.data));
});

evtSource.addEventListener('gate', (e) => {
  const data = JSON.parse(e.data);
  addEventToEntry(data.entryId, data);
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

evtSource.addEventListener('stats', (e) => {
  updateStats(JSON.parse(e.data));
});

evtSource.onerror = () => {
  console.log('SSE reconnecting...');
};
</script>
</body>
</html>`;
