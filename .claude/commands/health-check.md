# Codebase Health Check

Run a comprehensive health check on the listen project. This combines deterministic automated checks with AI-powered analysis of code quality, documentation freshness, and architectural consistency.

## Step 1: Run the automated health check script

Run `bun run scripts/health-check.ts` and capture the output. This checks:
- Tests pass
- Python/TypeScript config alignment (skills, tools, actions)
- Eval cases reference only valid skill.action pairs
- No stale naming references from past migrations
- Knowledge base docs match actual code
- Playlist infrastructure is intact
- Training data system prompts match config

If any checks fail, fix them before proceeding to Step 2.

## Step 2: AI-powered analysis

Use the `explore` subagent (thoroughness: very thorough) to investigate these areas that require semantic understanding beyond regex:

### 2a. Comment/docstring drift
Search for inline comments and docstrings across all `.ts`, `.py`, and `.md` files that reference specific skill names, tool names, dimension names, or architectural patterns. Flag any that describe behavior that no longer matches the actual code. Pay special attention to:
- JSDoc comments describing function parameters or return values
- Python docstrings with examples
- Inline comments explaining "why" that reference stale context

### 2b. Dead code detection
Look for:
- Exported functions/classes that have zero importers anywhere in the codebase
- Skill files that exist but are not registered in `DEFAULT_SKILLS` and have no active importers
- Test files that test functionality that has been removed or replaced
- Scripts in `scripts/` that reference removed functionality

### 2c. Knowledge base accuracy
For each file in `.knowledge/`:
- Check that architecture diagrams match actual file structure
- Check that version numbers, accuracy metrics, and training data counts match the actual artifacts
- Check that "Current State" and "Immediate Next Steps" sections are not stale
- Flag any section that references a file path that no longer exists

### 2d. Consistency across data boundaries
Check alignment across the Python/TypeScript/JSON boundary:
- Training data generator templates produce entries that match eval case expectations
- The HTTP server response format matches what the TypeScript client parses
- Skill hint regexes in TypeScript cover the same intent space as Python training positives

## Step 3: Report

Produce a structured report with:
- **Pass/Fail** from the automated script (Step 1)
- **Issues found** from AI analysis (Step 2), categorized as:
  - `[FIX]` — should be fixed now (stale comments, dead code, wrong docs)
  - `[TRACK]` — known tech debt to track but not urgent
  - `[OK]` — checked and clean
- **Summary stats**: X checks pass, Y issues found, Z items to track

If there are `[FIX]` items, offer to fix them. If the user agrees, fix them in a single commit.
