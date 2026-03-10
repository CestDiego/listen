/**
 * Watchlist — pattern matching against transcript chunks.
 *
 * Runs on EVERY chunk (not gated) for immediate detection.
 * Supports string triggers (case-insensitive fuzzy) and regex.
 * Has per-pattern cooldown to avoid spamming.
 */

import { readFile } from "fs/promises";
import { resolve } from "path";

// ── Types ──────────────────────────────────────────────────────────

export interface WatchlistResponse {
  /** macOS system sound name (e.g. "Purr", "Glass") */
  sound?: string;
  /** Text to speak via macOS `say` or ElevenLabs */
  voice?: string;
  /** macOS TTS voice name (default: "Samantha") */
  voiceName?: string;
  /** Notification body text */
  notification?: string;
}

export interface WatchlistPattern {
  id: string;
  category: string;
  severity: "low" | "medium" | "high" | "critical";
  triggers: string[];
  cooldownSeconds: number;
  response: WatchlistResponse;
}

export interface WatchlistConfig {
  patterns: WatchlistPattern[];
}

export interface WatchlistMatch {
  pattern: WatchlistPattern;
  trigger: string;
  matchedText: string;
  timestamp: Date;
}

// ── Matcher ────────────────────────────────────────────────────────

export class WatchlistMatcher {
  private patterns: WatchlistPattern[] = [];
  /** Last fire time per pattern id — for cooldown */
  private lastFired: Map<string, number> = new Map();

  /** Load patterns from a JSON file. */
  async load(filePath: string): Promise<number> {
    const resolvedPath = resolve(filePath);
    try {
      const raw = await readFile(resolvedPath, "utf-8");
      const config: WatchlistConfig = JSON.parse(raw);
      this.patterns = config.patterns || [];
      return this.patterns.length;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  ⚠ watchlist load failed: ${msg}`);
      return 0;
    }
  }

  /** Get count of loaded patterns. */
  get patternCount(): number {
    return this.patterns.length;
  }

  /**
   * Check a transcript chunk against all patterns.
   * Returns all matches (respecting cooldown).
   */
  check(text: string): WatchlistMatch[] {
    if (!text || this.patterns.length === 0) return [];

    const now = Date.now();
    const normalized = text.toLowerCase();
    const matches: WatchlistMatch[] = [];

    for (const pattern of this.patterns) {
      // Cooldown check
      const lastTime = this.lastFired.get(pattern.id) ?? 0;
      if (now - lastTime < pattern.cooldownSeconds * 1000) {
        continue; // still in cooldown
      }

      for (const trigger of pattern.triggers) {
        const isRegex =
          trigger.startsWith("/") && trigger.lastIndexOf("/") > 0;

        let matched = false;
        let matchedText = "";

        if (isRegex) {
          // Extract regex pattern between first and last /
          const lastSlash = trigger.lastIndexOf("/");
          const regexBody = trigger.slice(1, lastSlash);
          const regexFlags = trigger.slice(lastSlash + 1) || "i";
          try {
            const re = new RegExp(regexBody, regexFlags);
            const m = normalized.match(re);
            if (m) {
              matched = true;
              matchedText = m[0];
            }
          } catch {
            // Invalid regex — skip
          }
        } else {
          // Case-insensitive substring match
          const triggerLower = trigger.toLowerCase();
          if (normalized.includes(triggerLower)) {
            matched = true;
            matchedText = trigger;
          }
        }

        if (matched) {
          this.lastFired.set(pattern.id, now);
          matches.push({
            pattern,
            trigger,
            matchedText,
            timestamp: new Date(),
          });
          break; // one match per pattern is enough
        }
      }
    }

    return matches;
  }

  /** Get summary of loaded patterns by category. */
  summary(): Record<string, number> {
    const counts: Record<string, number> = {};
    for (const p of this.patterns) {
      counts[p.category] = (counts[p.category] || 0) + 1;
    }
    return counts;
  }
}
