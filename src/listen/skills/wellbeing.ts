/**
 * Wellbeing Skill — detects negative self-talk, burnout, and self-doubt.
 *
 * Wraps the existing watchlist system as a self-describing skill.
 * The LLM router decides WHEN to fire (catches things regex can't).
 * The watchlist JSON defines HOW to respond (sound, voice, notification).
 *
 * When the router triggers "check_in", this skill:
 *   1. Finds the best matching watchlist pattern (for response config)
 *   2. Falls back to a default gentle response if no pattern matches
 *   3. Respects per-pattern cooldowns
 */

import { readFile } from "fs/promises";
import { resolve } from "path";
import type { Skill, SkillResponse, RouterContext } from "./types";

// ── Watchlist types (from existing watchlist.ts) ──────────────────

interface WatchlistResponse {
  sound?: string;
  voice?: string;
  voiceName?: string;
  notification?: string;
}

interface WatchlistPattern {
  id: string;
  category: string;
  severity: "low" | "medium" | "high" | "critical";
  triggers: string[];
  cooldownSeconds: number;
  response: WatchlistResponse;
}

// ── State ─────────────────────────────────────────────────────────

let patterns: WatchlistPattern[] = [];
const lastFired: Map<string, number> = new Map();

// ── Default responses (when no watchlist pattern matches) ─────────

const DEFAULT_RESPONSES: Record<string, WatchlistResponse> = {
  "negative-self-talk": {
    sound: "Purr",
    voice: "Remember — you are valued, you are enough, and you are loved.",
    notification: "You matter. You are enough.",
  },
  burnout: {
    sound: "Submarine",
    voice: "It's okay to pause. Step away for five minutes — you'll come back sharper.",
    notification: "Take 5. Step away. Breathe.",
  },
  "self-doubt": {
    sound: "Glass",
    voice: "Hey — setbacks are part of growth. You're doing better than you think.",
    notification: "Setbacks are part of growth.",
  },
  default: {
    sound: "Purr",
    voice: "I noticed something. Just checking in — you're doing okay.",
    notification: "Checking in.",
  },
};

// ── Pattern matching ──────────────────────────────────────────────

/**
 * Find the best matching watchlist pattern for a transcript.
 * Uses the same string/regex matching as the original watchlist.ts.
 */
function findBestPattern(
  text: string
): { pattern: WatchlistPattern; trigger: string } | null {
  if (!text || patterns.length === 0) return null;

  const normalized = text.toLowerCase();
  const now = Date.now();

  for (const pattern of patterns) {
    // Cooldown check
    const lastTime = lastFired.get(pattern.id) ?? 0;
    if (now - lastTime < pattern.cooldownSeconds * 1000) continue;

    for (const trigger of pattern.triggers) {
      const isRegex = trigger.startsWith("/") && trigger.lastIndexOf("/") > 0;

      if (isRegex) {
        const lastSlash = trigger.lastIndexOf("/");
        const body = trigger.slice(1, lastSlash);
        const flags = trigger.slice(lastSlash + 1) || "i";
        try {
          if (new RegExp(body, flags).test(normalized)) {
            lastFired.set(pattern.id, now);
            return { pattern, trigger };
          }
        } catch {
          /* invalid regex */
        }
      } else {
        if (normalized.includes(trigger.toLowerCase())) {
          lastFired.set(pattern.id, now);
          return { pattern, trigger };
        }
      }
    }
  }

  return null;
}

// ── Skill definition ──────────────────────────────────────────────

export const wellbeingSkill: Skill = {
  name: "wellbeing",
  description:
    "Detects negative self-talk, self-doubt, burnout signals, and imposter syndrome. " +
    "Responds with gentle sounds, soothing voice messages, and supportive notifications. " +
    "Only activate when the speaker is talking ABOUT THEMSELVES negatively, not discussing others.",

  actions: [
    {
      name: "check_in",
      description:
        "Gently check in when detecting self-criticism, despair, burnout, or imposter feelings",
      params: [
        {
          name: "category",
          description:
            'Type of signal: "negative-self-talk", "self-doubt", "burnout", "imposter-syndrome"',
          required: false,
        },
      ],
    },
  ],

  hints: [
    /\bi\s+(hate|am|'m)\s+(myself|worthless|stupid|a\s+failure|a\s+fraud|so\s+tired)/i,
    /\b(can't\s+do\s+anything|give\s+up|don't\s+belong|burning\s+out|want\s+to\s+die)/i,
    /\b(everyone\s+hates|nobody\s+loves|not\s+good\s+enough|not\s+smart\s+enough)/i,
    /\b(i\s+always\s+(mess|fail|screw))/i,
  ],

  async init() {
    // Try loading watchlist patterns
    const paths = ["watchlist.json", "watchlist.default.json"];
    for (const p of paths) {
      try {
        const raw = await readFile(resolve(p), "utf-8");
        const config = JSON.parse(raw);
        // Only load wellbeing-category patterns
        patterns = (config.patterns || []).filter(
          (pat: WatchlistPattern) =>
            pat.category === "wellbeing" || pat.category === "communication"
        );
        if (patterns.length > 0) {
          console.log(
            `  🫀 wellbeing: loaded ${patterns.length} patterns from ${p}`
          );
          return;
        }
      } catch {
        /* try next */
      }
    }
    console.log("  🫀 wellbeing: using default responses (no watchlist file)");
  },

  async handle(
    action: string,
    params: Record<string, string>,
    ctx: RouterContext
  ): Promise<SkillResponse> {
    if (action !== "check_in") {
      return { success: false };
    }

    // Try to find a specific pattern match for the response config
    const match = findBestPattern(ctx.transcript);

    let response: WatchlistResponse;

    if (match) {
      response = match.pattern.response;
      console.log(
        `  🫀 wellbeing: matched pattern "${match.pattern.id}" via trigger "${match.trigger}"`
      );
    } else {
      // Use category-based default, or generic
      const category = params.category || "default";
      response =
        DEFAULT_RESPONSES[category] || DEFAULT_RESPONSES["default"];
      console.log(
        `  🫀 wellbeing: using default response for category "${category}"`
      );
    }

    return {
      success: true,
      sound: response.sound,
      voice: response.voice,
      notification: response.notification,
    };
  },
};
