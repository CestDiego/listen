/**
 * Transcriber — runs mlx_whisper on audio chunks.
 *
 * Uses Apple Silicon MLX backend for fast local inference.
 * Uses Bun.spawn with proper arg arrays to avoid shell issues.
 */

import type { ListenConfig, TranscriptChunk } from "./config";

/**
 * Known Whisper hallucinations on silence/near-silence.
 * These get output when the model has no real speech to transcribe.
 */
const HALLUCINATION_PHRASES = new Set([
  "thank you.",
  "thank you",
  "thanks for watching.",
  "thanks for watching",
  "subscribe to my channel.",
  "please subscribe.",
  "like and subscribe.",
  "thank you for watching.",
  "thanks.",
  "you",
  "bye.",
  "bye",
  "the end.",
  "...",
  ".",
  "",
]);

/**
 * Transcribe an audio file using mlx_whisper.
 * Returns a TranscriptChunk with the text and metadata.
 * Filters known Whisper hallucination phrases.
 */
export async function transcribe(
  audioPath: string,
  config: ListenConfig
): Promise<TranscriptChunk> {
  const timestamp = new Date();
  const outputDir = config.tmpDir;
  const baseName = audioPath.replace(/\.wav$/, "");

  // Build args as proper array — no string interpolation issues
  const args = [
    "mlx_whisper",
    "--model",
    config.whisperModel,
    "--output-dir",
    outputDir,
    "--output-format",
    "txt",
    ...(config.language ? ["--language", config.language] : []),
    audioPath,
  ];

  const proc = Bun.spawn(args, {
    stdout: "pipe",
    stderr: "pipe",
  });

  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    // Whisper can fail on silence — return empty chunk
    if (config.verbose) {
      const stderr = await new Response(proc.stderr).text();
      if (stderr.trim()) {
        console.error(`  ⚠ whisper stderr: ${stderr.slice(0, 100)}`);
      }
    }
    return { text: "", timestamp, durationSeconds: config.chunkSeconds };
  }

  // Read the transcription output
  const txtPath = `${baseName}.txt`;
  let text = "";
  try {
    text = await Bun.file(txtPath).text();
    try {
      await Bun.file(txtPath).unlink();
    } catch {}
  } catch {
    // File might not exist if audio was pure silence
    text = "";
  }

  const trimmed = text.trim();

  // Filter Whisper hallucinations (common on silence)
  if (HALLUCINATION_PHRASES.has(trimmed.toLowerCase())) {
    return { text: "", timestamp, durationSeconds: config.chunkSeconds };
  }

  return {
    text: trimmed,
    timestamp,
    durationSeconds: config.chunkSeconds,
  };
}
