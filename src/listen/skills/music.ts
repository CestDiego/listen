/**
 * Music Skill — controls Apple Music via AppleScript.
 *
 * Actions: play, pause, resume, skip, previous, volume_up, volume_down
 *
 * Speaks back what it did ("Now playing", "Skipping track", etc.)
 * using the existing ElevenLabs / macOS say responder.
 */

import type { Skill, SkillResponse, RouterContext } from "./types";

// ── AppleScript helpers ───────────────────────────────────────────

async function osascript(script: string): Promise<string> {
  const proc = Bun.spawn(["osascript", "-e", script], {
    stdout: "pipe",
    stderr: "pipe",
  });

  const output = await new Response(proc.stdout).text();
  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    throw new Error(`osascript failed (${exitCode}): ${stderr.trim()}`);
  }

  return output.trim();
}

/** Check if Apple Music is running. */
async function isMusicRunning(): Promise<boolean> {
  try {
    const result = await osascript(
      'tell application "System Events" to (name of processes) contains "Music"'
    );
    return result === "true";
  } catch {
    return false;
  }
}

/** Get the current track info (single osascript call, delimiter-safe). */
async function getCurrentTrack(): Promise<{
  name: string;
  artist: string;
  album: string;
} | null> {
  try {
    const result = await osascript(
      'tell application "Music" to (name of current track) & "|||" & (artist of current track) & "|||" & (album of current track)'
    );
    const [name, artist, album] = result.split("|||");
    return { name: name || "Unknown", artist: artist || "Unknown", album: album || "" };
  } catch {
    return null;
  }
}

/** Get current player state. */
async function getPlayerState(): Promise<string> {
  try {
    return await osascript(
      'tell application "Music" to get player state as string'
    );
  } catch {
    return "unknown";
  }
}

// ── AppleScript string escaping ───────────────────────────────────

function escapeAS(s: string): string {
  return s
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\r/g, "\\r")
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t")
    .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, ""); // strip control chars
}

// ── MusicKit integration (via Swift menu bar app) ─────────────────
//
// The Swift app runs a MusicKit HTTP server on port 3839.
// For catalog tracks, we call it instead of using AppleScript + UI scripting.
// Flow: iTunes Search API (free, fast) → get catalogId → POST to Swift app → MusicKit plays.

const MUSICKIT_ENDPOINT = "http://localhost:3839";
const MUSICKIT_TIMEOUT_MS = 5_000;

interface MusicKitPlayResult {
  success: boolean;
  track: string;
  artist: string;
  album: string;
  catalogId: string;
}

/**
 * Play a song via MusicKit (through the Swift menu bar app).
 * Accepts either a catalogId (fast) or a search query (search + play).
 * Returns track info on success, null on failure.
 */
async function playViaMusicKit(
  opts: { catalogId?: string; query?: string }
): Promise<MusicKitPlayResult | null> {
  try {
    const res = await fetch(`${MUSICKIT_ENDPOINT}/api/music/play`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts),
      signal: AbortSignal.timeout(MUSICKIT_TIMEOUT_MS),
    });

    if (!res.ok) {
      const err = await res.text();
      console.error(`  ⚠ MusicKit play failed (${res.status}): ${err}`);
      return null;
    }

    return (await res.json()) as MusicKitPlayResult;
  } catch (err) {
    // MusicKit server not running — fall back gracefully
    console.log(`  ⚠ MusicKit unavailable: ${err instanceof Error ? err.message : err}`);
    return null;
  }
}

/**
 * Check if the MusicKit server is reachable.
 */
async function isMusicKitAvailable(): Promise<boolean> {
  try {
    const res = await fetch(`${MUSICKIT_ENDPOINT}/api/music/status`, {
      signal: AbortSignal.timeout(1_000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Search the Apple Music streaming catalog via the iTunes Search API.
 * Free, no auth, ~150-600ms. Returns the top match or null.
 */
async function searchCatalog(
  query: string
): Promise<{ trackName: string; artistName: string; trackId: string } | null> {
  try {
    const url = `https://itunes.apple.com/search?${new URLSearchParams({
      term: query,
      entity: "song",
      limit: "1",
      media: "music",
    })}`;

    const res = await fetch(url, { signal: AbortSignal.timeout(3_000) });
    if (!res.ok) return null;

    const data = (await res.json()) as {
      resultCount: number;
      results: { trackName: string; artistName: string; trackId: number }[];
    };

    if (data.resultCount === 0) return null;
    const r = data.results[0];
    return { trackName: r.trackName, artistName: r.artistName, trackId: String(r.trackId) };
  } catch {
    return null;
  }
}

// ── Action handlers ───────────────────────────────────────────────
// Every handler checks the current player state BEFORE acting.
// If the action is redundant (e.g. pausing already-paused music),
// it returns success: true but skips the command and gives smart feedback.

async function handlePlay(
  params: Record<string, string>
): Promise<SkillResponse> {
  const song = params.song;
  const artist = params.artist;

  if (song) {
    const searchQuery = artist ? `${song} ${artist}` : song;

    // ── Step 1: Try local library first (fast, ~200ms) ────────
    try {
      const countResult = await osascript(
        `tell application "Music"
          activate
          set searchResults to search playlist "Library" for "${escapeAS(searchQuery)}"
          if (count of searchResults) > 0 then
            play item 1 of searchResults
            return "found"
          else
            return "not_found"
          end if
        end tell`
      );

      if (countResult === "found") {
        const track = await getCurrentTrack();
        return {
          success: true,
          voice: track
            ? `Playing ${track.name} by ${track.artist}`
            : `Playing ${searchQuery}`,
          sound: "Pop",
        };
      }
    } catch {
      // Library search failed — fall through to catalog
    }

    // ── Step 2: Search Apple Music catalog + play via MusicKit ──
    console.log(`  🔍 "${searchQuery}" not in library, searching catalog...`);

    // Try MusicKit first (direct search + play, ~1-2s)
    const mkResult = await playViaMusicKit({ query: searchQuery });
    if (mkResult?.success) {
      return {
        success: true,
        voice: `Playing ${mkResult.track} by ${mkResult.artist}`,
        sound: "Pop",
      };
    }

    // MusicKit unavailable — try iTunes Search API + MusicKit by catalogId
    const catalogResult = await searchCatalog(searchQuery);
    if (catalogResult) {
      console.log(`  🎵 catalog hit: ${catalogResult.trackName} by ${catalogResult.artistName}`);
      const mkById = await playViaMusicKit({ catalogId: catalogResult.trackId });
      if (mkById?.success) {
        return {
          success: true,
          voice: `Playing ${mkById.track} by ${mkById.artist}`,
          sound: "Pop",
        };
      }
    }

    // ── Step 3: Nothing found anywhere — just resume playback ──
    await osascript('tell application "Music" to play');
    return {
      success: true,
      voice: `I couldn't find "${searchQuery}" in your library or Apple Music.`,
    };
  }

  // Generic play (no specific song) — check if already playing
  const state = await getPlayerState();
  if (state === "playing") {
    const track = await getCurrentTrack();
    return {
      success: true,
      voice: track
        ? `Already playing ${track.name}`
        : "Music is already playing",
    };
  }

  await osascript('tell application "Music" to play');
  const track = await getCurrentTrack();
  return {
    success: true,
    voice: track
      ? `Playing ${track.name} by ${track.artist}`
      : "Music is playing",
    sound: "Pop",
  };
}

async function handlePause(): Promise<SkillResponse> {
  const state = await getPlayerState();
  if (state === "paused" || state === "stopped") {
    return {
      success: true,
      voice: "Music is already paused",
    };
  }

  await osascript('tell application "Music" to pause');
  return {
    success: true,
    voice: "Music paused",
    sound: "Tink",
  };
}

async function handleResume(): Promise<SkillResponse> {
  const state = await getPlayerState();
  if (state === "playing") {
    const track = await getCurrentTrack();
    return {
      success: true,
      voice: track
        ? `Already playing ${track.name}`
        : "Music is already playing",
    };
  }

  await osascript('tell application "Music" to play');
  const track = await getCurrentTrack();
  return {
    success: true,
    voice: track ? `Resuming ${track.name}` : "Resuming playback",
    sound: "Pop",
  };
}

async function handleSkip(): Promise<SkillResponse> {
  const state = await getPlayerState();
  if (state === "stopped" || state === "paused") {
    return {
      success: true,
      voice: "Nothing is playing right now",
    };
  }

  await osascript('tell application "Music" to next track');
  await new Promise((r) => setTimeout(r, 500));
  const track = await getCurrentTrack();
  return {
    success: true,
    voice: track ? `Now playing ${track.name} by ${track.artist}` : "Skipped to next track",
    sound: "Pop",
  };
}

async function handlePrevious(): Promise<SkillResponse> {
  const state = await getPlayerState();
  if (state === "stopped" || state === "paused") {
    return {
      success: true,
      voice: "Nothing is playing right now",
    };
  }

  await osascript('tell application "Music" to previous track');
  await new Promise((r) => setTimeout(r, 500));
  const track = await getCurrentTrack();
  return {
    success: true,
    voice: track ? `Now playing ${track.name} by ${track.artist}` : "Previous track",
    sound: "Pop",
  };
}

async function handleVolumeUp(): Promise<SkillResponse> {
  const current = await osascript(
    'tell application "Music" to get sound volume'
  );
  const newVol = Math.min(100, Number(current) + 15);
  await osascript(
    `tell application "Music" to set sound volume to ${newVol}`
  );
  return {
    success: true,
    voice: `Volume up to ${newVol} percent`,
    sound: "Blow",
  };
}

async function handleVolumeDown(): Promise<SkillResponse> {
  const current = await osascript(
    'tell application "Music" to get sound volume'
  );
  const newVol = Math.max(0, Number(current) - 15);
  await osascript(
    `tell application "Music" to set sound volume to ${newVol}`
  );
  return {
    success: true,
    voice: `Volume down to ${newVol} percent`,
    sound: "Blow",
  };
}

// ── Skill definition ──────────────────────────────────────────────

export const musicSkill: Skill = {
  name: "music",
  description:
    "Controls Apple Music playback. Can search and play any song on Apple Music, not just the user's library. " +
    "Activate when the user gives a clear command to play, pause, skip, or adjust music. " +
    'Examples: "play some music", "play Bohemian Rhapsody", "skip this song", "turn it up". ' +
    "Do NOT activate when the user is merely talking ABOUT music.",

  async getState(): Promise<Record<string, string>> {
    const running = await isMusicRunning();
    if (!running) return { status: "not running" };

    const state = await getPlayerState();
    const track = await getCurrentTrack();
    const result: Record<string, string> = { status: state };
    if (track) {
      result.track = `${track.name} by ${track.artist}`;
    }
    return result;
  },

  actions: [
    {
      name: "play",
      description: "Start playing music. Can search by song/artist — tries local library first, then Apple Music catalog",
      params: [
        { name: "song", description: "Song name to search for", required: false },
        { name: "artist", description: "Artist name to search for", required: false },
      ],
    },
    {
      name: "pause",
      description: "Pause playback",
    },
    {
      name: "resume",
      description: "Resume paused playback",
    },
    {
      name: "skip",
      description: "Skip to the next track",
    },
    {
      name: "previous",
      description: "Go back to the previous track",
    },
    {
      name: "volume_up",
      description: "Increase the volume",
    },
    {
      name: "volume_down",
      description: "Decrease the volume",
    },
  ],

  hints: [
    /\b(play|pause|resume|skip|next|previous|stop)\b.*\b(music|song|track|album)\b/i,
    /\b(play|pause|skip|next)\s+(some\s+)?music\b/i,
    /\b(turn|volume)\s+(up|down|it\s+up|it\s+down)\b/i,
    /\bplay\s+(me\s+)?some\b/i,
    /\bskip\s+(this|the)\s+(song|track)\b/i,
    /\bnext\s+(song|track)\b/i,
    /\bpause\s+(it|that|the\s+music)\b/i,
  ],

  async init() {
    const running = await isMusicRunning();
    console.log(
      `  🎵 music: Apple Music ${running ? "is running" : "not running (will launch on first command)"}`
    );
  },

  async handle(
    action: string,
    params: Record<string, string>,
    _ctx: RouterContext
  ): Promise<SkillResponse> {
    // Ensure Music is at least launched
    const running = await isMusicRunning();
    if (!running && action !== "pause") {
      await osascript('tell application "Music" to activate');
      await new Promise((r) => setTimeout(r, 1000));
    }

    switch (action) {
      case "play":
        return handlePlay(params);
      case "pause":
        return handlePause();
      case "resume":
        return handleResume();
      case "skip":
        return handleSkip();
      case "previous":
        return handlePrevious();
      case "volume_up":
        return handleVolumeUp();
      case "volume_down":
        return handleVolumeDown();
      default:
        return { success: false, voice: `Unknown music action: ${action}` };
    }
  },
};
