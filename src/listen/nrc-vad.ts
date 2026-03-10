/**
 * NRC VAD Lexicon loader — word-level Valence/Arousal/Dominance scores.
 *
 * Loads the NRC-VAD-Lexicon-v2.1 unigrams file (44,728 words) into a Map
 * for O(1) per-word lookup. Used by the mood (valence) and energy (arousal)
 * computed dimensions in intent-vector.ts.
 *
 * Lexicon: https://saifmohammad.com/WebPages/nrc-vad.html
 * Scale: [-1, +1] for all three dimensions (v2.1 centered at 0).
 *
 * The file is NOT redistributed (license restriction) — download manually
 * or via the setup script. If the file is missing, a warning is logged
 * and an empty lexicon is returned (graceful degradation to the old heuristic).
 */

import { existsSync, readFileSync } from "fs";
import { resolve } from "path";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VADScores {
  /** Valence: pleasure/displeasure [-1, +1]. Positive = pleasant. */
  valence: number;
  /** Arousal: excited/calm [-1, +1]. Positive = high energy. */
  arousal: number;
  /** Dominance: powerful/weak [-1, +1]. Positive = in control. */
  dominance: number;
}

// ---------------------------------------------------------------------------
// Lexicon singleton
// ---------------------------------------------------------------------------

let lexicon: Map<string, VADScores> | null = null;
let loadAttempted = false;

/** Default path relative to project root. */
const DEFAULT_PATH = resolve(
  import.meta.dir,
  "../../data/nrc-vad/unigrams-NRC-VAD-Lexicon-v2.1.txt",
);

/**
 * Load the NRC VAD lexicon (once). Returns the Map on success,
 * or an empty Map if the file is missing (with a console warning).
 */
export function loadLexicon(filePath?: string): Map<string, VADScores> {
  if (lexicon !== null) return lexicon;
  loadAttempted = true;

  const path = filePath ?? DEFAULT_PATH;

  if (!existsSync(path)) {
    console.warn(
      `  ⚠ NRC VAD Lexicon not found at ${path}` +
      `\n    Mood/energy will use fallback heuristics.` +
      `\n    Download from: https://saifmohammad.com/WebPages/nrc-vad.html`,
    );
    lexicon = new Map();
    return lexicon;
  }

  // Sync read — lexicon loads once at startup, before the event loop matters.
  const content = readFileSync(path, "utf-8");
  const lines = content.split("\n");

  lexicon = new Map();
  // Skip header line ("term\tvalence\tarousal\tdominance")
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line) continue;
    const tab1 = line.indexOf("\t");
    const tab2 = line.indexOf("\t", tab1 + 1);
    const tab3 = line.indexOf("\t", tab2 + 1);
    if (tab1 < 0 || tab2 < 0 || tab3 < 0) continue;

    const term = line.slice(0, tab1);
    const v = parseFloat(line.slice(tab1 + 1, tab2));
    const a = parseFloat(line.slice(tab2 + 1, tab3));
    const d = parseFloat(line.slice(tab3 + 1));

    if (term && !isNaN(v) && !isNaN(a) && !isNaN(d)) {
      lexicon.set(term, { valence: v, arousal: a, dominance: d });
    }
  }

  console.log(`  📖 NRC VAD Lexicon: ${lexicon.size} words loaded`);
  return lexicon;
}

/** Check if the lexicon is loaded and non-empty. */
export function isLexiconAvailable(): boolean {
  if (lexicon === null && !loadAttempted) loadLexicon();
  return lexicon !== null && lexicon.size > 0;
}

/** Look up a single word. Returns undefined if not in lexicon. */
export function lookupWord(word: string): VADScores | undefined {
  if (lexicon === null && !loadAttempted) loadLexicon();
  return lexicon?.get(word.toLowerCase());
}

// ---------------------------------------------------------------------------
// Aggregate scoring functions for transcript chunks
// ---------------------------------------------------------------------------

/**
 * Compute average valence for a transcript chunk.
 * Returns 0 if no lexicon words found. Range: [-1, +1].
 */
export function chunkValence(text: string): number {
  if (!isLexiconAvailable()) return 0;
  const words = tokenize(text);
  let sum = 0;
  let count = 0;
  for (const w of words) {
    const entry = lexicon!.get(w);
    if (entry) {
      sum += entry.valence;
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/**
 * Compute average arousal for a transcript chunk.
 * Returns 0 if no lexicon words found. Range: [-1, +1].
 */
export function chunkArousal(text: string): number {
  if (!isLexiconAvailable()) return 0;
  const words = tokenize(text);
  let sum = 0;
  let count = 0;
  for (const w of words) {
    const entry = lexicon!.get(w);
    if (entry) {
      sum += entry.arousal;
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/**
 * Compute combined VAD averages for a transcript chunk.
 * Returns all three dimensions at once (avoids double iteration).
 */
export function chunkVAD(text: string): { valence: number; arousal: number; dominance: number; coverage: number } {
  if (!isLexiconAvailable()) return { valence: 0, arousal: 0, dominance: 0, coverage: 0 };
  const words = tokenize(text);
  let vSum = 0;
  let aSum = 0;
  let dSum = 0;
  let count = 0;
  for (const w of words) {
    const entry = lexicon!.get(w);
    if (entry) {
      vSum += entry.valence;
      aSum += entry.arousal;
      dSum += entry.dominance;
      count++;
    }
  }
  const total = words.length || 1;
  return {
    valence: count > 0 ? vSum / count : 0,
    arousal: count > 0 ? aSum / count : 0,
    dominance: count > 0 ? dSum / count : 0,
    coverage: count / total, // fraction of words found in lexicon
  };
}

// ---------------------------------------------------------------------------
// Tokenizer (simple, shared)
// ---------------------------------------------------------------------------

/** Lowercase, split on whitespace, strip trailing punctuation. */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/\s+/)
    .map((w) => w.replace(/[.,!?;:'"]+$/g, "").replace(/^['"]+/g, ""))
    .filter(Boolean);
}
