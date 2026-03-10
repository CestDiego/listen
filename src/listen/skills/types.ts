/**
 * Skill system types.
 *
 * A skill is a self-describing module that:
 *   - declares what it can do (actions + descriptions)
 *   - handles actions when the router triggers them
 *   - optionally provides pre-filter hints for optimization
 *
 * The router uses skill descriptions to build a single LLM prompt
 * that classifies transcripts across ALL skills in one call.
 */

// ── Skill definition ──────────────────────────────────────────────

export interface SkillParam {
  name: string;
  description: string;
  required?: boolean;
}

export interface SkillAction {
  name: string;
  description: string;
  params?: SkillParam[];
}

export interface SkillResponse {
  /** Did the action succeed? */
  success: boolean;
  /** Optional voice feedback (spoken via ElevenLabs / macOS say) */
  voice?: string;
  /** Optional macOS system sound to play */
  sound?: string;
  /** Optional macOS notification */
  notification?: string;
}

export interface Skill {
  /** Unique identifier (e.g. "music", "wellbeing") */
  name: string;

  /** Human description for the LLM router prompt */
  description: string;

  /** Available actions this skill can perform */
  actions: SkillAction[];

  /**
   * Optional regex hints for pre-filtering.
   * If provided AND no hints match the transcript, the router
   * may skip the LLM call for this skill (optimization).
   * If omitted, the skill is always included in routing.
   */
  hints?: RegExp[];

  /**
   * Called when the router decides this skill should handle an action.
   */
  handle: (
    action: string,
    params: Record<string, string>,
    ctx: RouterContext
  ) => Promise<SkillResponse>;

  /**
   * Optional lifecycle: called once when the skill is registered.
   * Use for loading config files, checking prerequisites, etc.
   */
  init?: () => Promise<void>;
}

// ── Router types ──────────────────────────────────────────────────

export interface SkillMatch {
  /** Which skill to activate */
  skill: string;
  /** Which action to run */
  action: string;
  /** Extracted parameters */
  params: Record<string, string>;
  /** Router's confidence (0.0 - 1.0) */
  confidence: number;
}

export interface RouterResult {
  /** Skills that should fire */
  matches: SkillMatch[];
  /** General interest score 0-10 (replaces old gate) */
  interest: number;
  /** Brief explanation */
  reason: string;
}

/** Record of a skill that recently executed (for router memory). */
export interface SkillExecution {
  skill: string;
  action: string;
  success: boolean;
  voice?: string;
  timestamp: Date;
}

export interface RouterContext {
  /** The transcript chunk that triggered routing */
  transcript: string;
  /** Rolling buffer context (recent conversation) */
  buffer: string;
  /** Timestamp of the transcript */
  timestamp: Date;
  /** Router's reasoning for why this skill was triggered (set after routing) */
  routerReason?: string;
  /** All skills the router matched this cycle (set after routing, before handlers run) */
  allMatches?: SkillMatch[];
  /** Recent skill executions (last N, for handler awareness + router memory) */
  recentSkills?: SkillExecution[];
}
