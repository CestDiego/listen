/**
 * Skill Registry — manages loaded skills and builds the router prompt.
 *
 * Skills self-describe their actions and parameters.
 * The registry composes these descriptions into a single prompt
 * that the LLM uses to route transcripts to the right skill(s).
 */

import type { Skill, SkillMatch, SkillResponse, RouterContext } from "./types";

// ── Registry ──────────────────────────────────────────────────────

/** Default cooldown per skill.action to prevent feedback loops (ms). */
const SKILL_COOLDOWN_MS = 8_000;

export class SkillRegistry {
  private skills: Map<string, Skill> = new Map();

  /** Tracks last execution time per "skill.action" key. */
  private lastExecution: Map<string, number> = new Map();

  /** Register a skill. Calls skill.init() if defined. */
  async register(skill: Skill): Promise<void> {
    if (this.skills.has(skill.name)) {
      console.warn(`  ⚠ skill "${skill.name}" already registered, replacing`);
    }
    if (skill.init) {
      await skill.init();
    }
    this.skills.set(skill.name, skill);
  }

  /** Get a registered skill by name. */
  get(name: string): Skill | undefined {
    return this.skills.get(name);
  }

  /** Get all registered skills. */
  all(): Skill[] {
    return Array.from(this.skills.values());
  }

  /** Number of registered skills. */
  get size(): number {
    return this.skills.size;
  }

  /**
   * Build the skills description block for the router prompt.
   * This is what the LLM sees to understand available skills.
   */
  buildSkillsPrompt(): string {
    const lines: string[] = [];

    for (const skill of this.skills.values()) {
      lines.push(`- ${skill.name}: ${skill.description}`);

      for (const action of skill.actions) {
        const paramList = action.params?.length
          ? action.params.map((p) => (p.required ? p.name : `${p.name}?`)).join(", ")
          : "";
        lines.push(`    ${action.name}(${paramList}): ${action.description}`);
      }
    }

    return lines.join("\n");
  }

  /**
   * Query the current state of all skills that expose getState().
   * Returns a formatted string for the router prompt, e.g.:
   *   accommodator: status=matching, quadrant=calm
   */
  async buildStateContext(): Promise<string> {
    const lines: string[] = [];

    for (const skill of this.skills.values()) {
      if (!skill.getState) continue;
      try {
        const state = await skill.getState();
        const pairs = Object.entries(state)
          .map(([k, v]) => `${k}=${v}`)
          .join(", ");
        lines.push(`  ${skill.name}: ${pairs}`);
      } catch {
        lines.push(`  ${skill.name}: state unavailable`);
      }
    }

    return lines.length > 0 ? lines.join("\n") : "";
  }

  /**
   * Execute a skill match — looks up the skill and calls its handler.
   * Returns the skill's response, or a failure response if the skill isn't found.
   */
  async execute(
    match: SkillMatch,
    ctx: RouterContext
  ): Promise<SkillResponse> {
    const skill = this.skills.get(match.skill);
    if (!skill) {
      console.error(`  ⚠ skill "${match.skill}" not found in registry`);
      return { success: false };
    }

    // ── Cooldown guard ─────────────────────────────────────────
    // Prevents the same skill.action from re-firing within the
    // cooldown window. This is the hard backstop against feedback
    // loops where the mic picks up the system's own voice response.
    const key = `${match.skill}.${match.action}`;
    const now = Date.now();
    const last = this.lastExecution.get(key);
    if (last && now - last < SKILL_COOLDOWN_MS) {
      const agoSec = ((now - last) / 1000).toFixed(1);
      console.log(`  ⏳ ${key} on cooldown (fired ${agoSec}s ago, window ${SKILL_COOLDOWN_MS / 1000}s)`);
      return { success: true, voice: undefined }; // swallow silently
    }

    try {
      const result = await skill.handle(match.action, match.params, ctx);
      this.lastExecution.set(key, Date.now()); // use completion time, not start time
      return result;
    } catch (err) {
      this.lastExecution.set(key, Date.now()); // cooldown even on failure to prevent loops
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  ⚠ skill "${match.skill}.${match.action}" failed: ${msg}`);
      return { success: false };
    }
  }

  /** Print a summary of registered skills for the banner. */
  summary(): string {
    return this.all()
      .map((s) => `${s.name}(${s.actions.length})`)
      .join(" ");
  }
}
