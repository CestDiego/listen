/**
 * Skills — pluggable action modules for the listen pipeline.
 *
 * Each skill self-describes its capabilities. The classifier broadcasts
 * transcripts to per-skill MLX experts in parallel via Promise.all.
 *
 * To add a new skill:
 *   1. Create a new file in this directory (e.g. notes.ts)
 *   2. Export a Skill object
 *   3. Add it to DEFAULT_SKILLS below
 *   4. Train an expert: cd experts && uv run generate <skill> && uv run train <skill>
 */

export type {
  Skill,
  SkillAction,
  SkillParam,
  SkillMatch,
  SkillResponse,
  SkillExecution,
  RouterResult,
  RouterContext,
  ExpertClassification,
  ClassifyResult,
} from "./types";

export { SkillRegistry } from "./registry";
export { classifyTranscript } from "./classify";
export { wellbeingSkill } from "./wellbeing";
export { musicSkill } from "./music";
export { accommodatorSkill, connectAccommodatorToIntentVector } from "./accommodator";

// ── Default skill set ─────────────────────────────────────────────

import type { Skill } from "./types";
import { wellbeingSkill } from "./wellbeing";
import { accommodatorSkill } from "./accommodator";

/**
 * All built-in skills. Register these with the SkillRegistry.
 * NOTE: musicSkill removed from DEFAULT_SKILLS — no longer a classifier target.
 * It remains importable for legacy/override use.
 */
export const DEFAULT_SKILLS: Skill[] = [wellbeingSkill, accommodatorSkill];
