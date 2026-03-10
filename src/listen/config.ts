/**
 * listen — live conversation monitor
 *
 * Configuration, types, and CLI argument parsing.
 */

import { parseArgs } from "util";

// ── Types ──────────────────────────────────────────────────────────

export interface ListenConfig {
  /** "live" = mic recording, "pipe" = read from stdin, "moonshine" = receive from Moonshine app */
  mode: "live" | "pipe" | "moonshine";

  /** ffmpeg avfoundation audio device index (macOS) */
  audioDevice: string;

  /** Duration of each audio chunk in seconds */
  chunkSeconds: number;

  /** Whisper model repo for mlx_whisper */
  whisperModel: string;

  /** Language hint for whisper (empty = auto-detect) */
  language: string;

  /** MLX expert server endpoint (per-skill fine-tuned models) */
  expertEndpoint: string;

  /** opencode model ID for the full analysis */
  analysisModel: string;

  /** Gate score threshold 0-10 — only escalate above this */
  threshold: number;

  /** How many chunks between gate checks */
  gateEveryNChunks: number;

  /** Max minutes of transcript to keep in the rolling buffer */
  bufferMinutes: number;

  /** Temp directory for audio chunks */
  tmpDir: string;

  /** Path to watchlist JSON (empty = disabled) */
  watchlistPath: string;

  /** Path to event log (JSONL) */
  eventLogPath: string;

  /** Show live transcript in terminal */
  verbose: boolean;
}

export const DEFAULT_CONFIG: ListenConfig = {
  mode: "live",
  audioDevice: ":0", // default audio input on macOS
  chunkSeconds: 5, // 5s chunks for faster reaction
  whisperModel: "mlx-community/whisper-large-v3-turbo",
  language: "",
  expertEndpoint: "http://localhost:8234", // MLX expert server
  analysisModel: "opencode/claude-sonnet-4-6",
  threshold: 6,
  gateEveryNChunks: 0, // 0 = gate every non-empty chunk (fast with local model)
  bufferMinutes: 5,
  tmpDir: "/tmp/listen-chunks",
  watchlistPath: "watchlist.default.json",
  eventLogPath: "/tmp/listen-events.jsonl",
  verbose: true,
};

// ── Gate result ────────────────────────────────────────────────────

export interface GateResult {
  score: number;
  reason: string;
}

// ── Analysis result ────────────────────────────────────────────────

export interface AnalysisResult {
  insights: string;
  timestamp: Date;
  triggerReason: string;
}

// ── Transcript chunk ───────────────────────────────────────────────

export interface TranscriptChunk {
  text: string;
  timestamp: Date;
  durationSeconds: number;
}

// ── Validation helpers ─────────────────────────────────────────────

function parsePositiveNum(
  val: string | undefined,
  fallback: number,
  name: string,
  allowZero = false
): number {
  const n = Number(val);
  if (!Number.isFinite(n) || (allowZero ? n < 0 : n <= 0)) {
    console.error(
      `  ⚠ invalid value for --${name}: "${val}", using default ${fallback}`
    );
    return fallback;
  }
  return n;
}

function parseThreshold(val: string | undefined): number {
  const n = Number(val);
  if (!Number.isFinite(n) || n < 0 || n > 10) {
    console.error(
      `  ⚠ threshold must be 0-10, using default ${DEFAULT_CONFIG.threshold}`
    );
    return DEFAULT_CONFIG.threshold;
  }
  return n;
}

// ── CLI arg parser ─────────────────────────────────────────────────

export function parseCliArgs(): ListenConfig {
  let values: Record<string, unknown>;

  try {
    const result = parseArgs({
      args: Bun.argv.slice(2),
      options: {
        pipe: { type: "boolean", default: false },
        moonshine: { type: "boolean", default: false },
        device: { type: "string", default: DEFAULT_CONFIG.audioDevice },
        chunk: {
          type: "string",
          default: String(DEFAULT_CONFIG.chunkSeconds),
        },
        "whisper-model": {
          type: "string",
          default: DEFAULT_CONFIG.whisperModel,
        },
        language: { type: "string", default: DEFAULT_CONFIG.language },
        "expert-endpoint": {
          type: "string",
          default: DEFAULT_CONFIG.expertEndpoint,
        },
        "analysis-model": {
          type: "string",
          default: DEFAULT_CONFIG.analysisModel,
        },
        threshold: {
          type: "string",
          default: String(DEFAULT_CONFIG.threshold),
        },
        "gate-every": {
          type: "string",
          default: String(DEFAULT_CONFIG.gateEveryNChunks),
        },
        "buffer-min": {
          type: "string",
          default: String(DEFAULT_CONFIG.bufferMinutes),
        },
        watchlist: {
          type: "string",
          default: DEFAULT_CONFIG.watchlistPath,
        },
        "no-watchlist": { type: "boolean", default: false },
        "event-log": {
          type: "string",
          default: DEFAULT_CONFIG.eventLogPath,
        },
        verbose: { type: "boolean", default: DEFAULT_CONFIG.verbose },
        quiet: { type: "boolean", default: false },
        help: { type: "boolean", short: "h", default: false },
      },
      strict: true,
    });
    values = result.values as Record<string, unknown>;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`  ✗ ${message}`);
    console.error(`  Run with --help for usage.`);
    process.exit(1);
  }

  if (values.help) {
    printHelp();
    process.exit(0);
  }

  return {
    ...DEFAULT_CONFIG,
    mode: values.moonshine ? "moonshine" : values.pipe ? "pipe" : "live",
    audioDevice: values.device as string,
    chunkSeconds: parsePositiveNum(
      values.chunk as string,
      DEFAULT_CONFIG.chunkSeconds,
      "chunk"
    ),
    whisperModel: values["whisper-model"] as string,
    language: values.language as string,
    expertEndpoint: values["expert-endpoint"] as string,
    analysisModel: values["analysis-model"] as string,
    threshold: parseThreshold(values.threshold as string),
    gateEveryNChunks: Math.round(
      parsePositiveNum(
        values["gate-every"] as string,
        DEFAULT_CONFIG.gateEveryNChunks,
        "gate-every",
        true // 0 = gate every chunk
      )
    ),
    bufferMinutes: parsePositiveNum(
      values["buffer-min"] as string,
      DEFAULT_CONFIG.bufferMinutes,
      "buffer-min"
    ),
    watchlistPath: values["no-watchlist"]
      ? ""
      : (values.watchlist as string),
    eventLogPath: values["event-log"] as string,
    verbose: values.quiet ? false : (values.verbose as boolean),
  };
}

function printHelp() {
  console.log(`
  ┌─────────────────────────────────┐
  │  listen — conversation monitor  │
  └─────────────────────────────────┘

  Records audio → transcribes → watches for patterns → 
  gates with cheap model → analyzes with big model →
  emits events + responds (sound, voice, notification).

  USAGE
    bun listen                    # live mic mode (default watchlist)
    bun listen --pipe             # read transcripts from stdin
    bun listen --moonshine        # receive from Moonshine menu bar app
    echo "text" | bun listen --pipe
    tail -f /tmp/listen-events.jsonl  # consume events from another process

  OPTIONS
    --pipe                  Read transcripts from stdin instead of mic
    --moonshine             Receive transcripts from Moonshine Swift app via POST
    --device <id>           ffmpeg avfoundation audio device (default: ":0")
    --chunk <seconds>       Audio chunk duration (default: 5)
    --whisper-model <repo>  mlx_whisper model (default: whisper-large-v3-turbo)
    --language <code>       Language hint for whisper (default: auto-detect)
    --expert-endpoint <url> MLX expert server (default: http://localhost:8234)
    --analysis-model <id>   Big model for analysis (default: opencode/claude-sonnet-4-6)
    --threshold <0-10>      Gate score to trigger analysis (default: 6)
    --gate-every <N>        Gate every N chunks, 0=every chunk (default: 0)
    --buffer-min <minutes>  Rolling buffer size (default: 5)
    --watchlist <path>      Watchlist JSON file (default: watchlist.default.json)
    --no-watchlist          Disable watchlist entirely
    --event-log <path>      Event log path (default: /tmp/listen-events.jsonl)
    --quiet                 Suppress live transcript output
    -h, --help              Show this help
  `);
}
