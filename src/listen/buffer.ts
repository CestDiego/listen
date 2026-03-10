/**
 * Sliding window transcript buffer.
 *
 * Keeps a rolling window of transcript chunks, evicting old ones
 * beyond the configured time limit. Tracks chunks since last gate check.
 */

import type { TranscriptChunk } from "./config";

export class TranscriptBuffer {
  private chunks: TranscriptChunk[] = [];
  private maxAgeMs: number;
  private _chunksSinceLastGate = 0;

  constructor(bufferMinutes: number) {
    this.maxAgeMs = bufferMinutes * 60 * 1000;
  }

  /** Append a new transcription chunk and bump the gate counter. */
  append(chunk: TranscriptChunk): void {
    this.chunks.push(chunk);
    this._chunksSinceLastGate++;
    this.evict();
  }

  /** Number of chunks received since the last gate reset. */
  get chunksSinceLastGate(): number {
    return this._chunksSinceLastGate;
  }

  /** Reset the gate counter (call after a gate check). */
  resetGateCounter(): void {
    this._chunksSinceLastGate = 0;
  }

  /** Get the most recent N minutes of text (for gate checks). */
  recentText(minutes?: number): string {
    const cutoff = minutes
      ? Date.now() - minutes * 60 * 1000
      : Date.now() - this.maxAgeMs;

    return this.chunks
      .filter((c) => c.timestamp.getTime() >= cutoff)
      .map((c) => c.text)
      .filter(Boolean)
      .join(" ")
      .trim();
  }

  /** Get the full rolling buffer text (for analysis). */
  fullContext(): string {
    this.evict();
    return this.chunks
      .map((c) => c.text)
      .filter(Boolean)
      .join(" ")
      .trim();
  }

  /** Get a timestamped version for analysis context. */
  fullContextTimestamped(): string {
    this.evict();
    return this.chunks
      .filter((c) => c.text.trim())
      .map((c) => {
        const t = c.timestamp.toLocaleTimeString("en-US", {
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        });
        return `[${t}] ${c.text}`;
      })
      .join("\n")
      .trim();
  }

  /** Total chunks in the buffer. */
  get size(): number {
    return this.chunks.length;
  }

  /** Approximate word count of the entire buffer (empty chunks = 0). */
  get wordCount(): number {
    return this.chunks.reduce(
      (sum, c) =>
        sum + (c.text.trim() ? c.text.trim().split(/\s+/).length : 0),
      0
    );
  }

  /** Remove chunks older than the max age. */
  private evict(): void {
    const cutoff = Date.now() - this.maxAgeMs;
    this.chunks = this.chunks.filter((c) => c.timestamp.getTime() >= cutoff);
  }
}
