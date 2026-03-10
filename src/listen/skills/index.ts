/**
 * Skills — pluggable action modules for the listen pipeline.
 *
 * Each skill self-describes its capabilities. The router uses these
 * descriptions to classify transcripts in a single LLM call.
 *
 * To add a new skill:
 *   1. Create a new file in this directory (e.g. notes.ts)
 *   2. Export a Skill object
 *   3. Add it to DEFAULT_SKILLS below
 *   4. The router prompt updates automatically
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
} from "./types";

export { SkillRegistry } from "./registry";
export { routeTranscript } from "./router";
export { wellbeingSkill } from "./wellbeing";
export { musicSkill } from "./music";

// ── Default skill set ─────────────────────────────────────────────

import type { Skill } from "./types";
import { wellbeingSkill } from "./wellbeing";
import { musicSkill } from "./music";

/** All built-in skills. Register these with the SkillRegistry. */
export const DEFAULT_SKILLS: Skill[] = [wellbeingSkill, musicSkill];
