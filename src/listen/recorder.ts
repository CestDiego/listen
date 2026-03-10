/**
 * Audio recorder — captures mic input via ffmpeg (avfoundation on macOS).
 *
 * Records fixed-duration WAV chunks to a temp directory.
 * Uses Bun.spawn with proper arg arrays.
 */

import { mkdir } from "fs/promises";
import type { ListenConfig } from "./config";

let chunkIndex = 0;

/** Ensure the temp directory exists. */
export async function initRecorder(config: ListenConfig): Promise<void> {
  await mkdir(config.tmpDir, { recursive: true });
}

/**
 * Record a single audio chunk from the mic.
 * Returns the path to the recorded WAV file.
 */
export async function recordChunk(config: ListenConfig): Promise<string> {
  const outPath = `${config.tmpDir}/chunk_${String(chunkIndex++).padStart(5, "0")}.wav`;

  // Use Bun.spawn with proper arg array for safety
  const proc = Bun.spawn(
    [
      "ffmpeg",
      "-f", "avfoundation",
      "-i", config.audioDevice,
      "-t", String(config.chunkSeconds),
      "-ar", "16000",
      "-ac", "1",
      "-y",
      outPath,
    ],
    {
      stdout: "pipe",
      stderr: "pipe",
    }
  );

  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    throw new Error(
      `ffmpeg recording failed (exit ${exitCode}): ${stderr.slice(0, 200)}\n` +
      `  hint: check --device flag or run: ffmpeg -f avfoundation -list_devices true -i ""`
    );
  }

  return outPath;
}

/** Clean up a chunk file after transcription. */
export async function cleanupChunk(path: string): Promise<void> {
  try {
    await Bun.file(path).unlink();
  } catch {
    // non-critical
  }
}
