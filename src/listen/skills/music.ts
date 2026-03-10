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

/** Get the current track info (single osascript call). */
async function getCurrentTrack(): Promise<{
  name: string;
  artist: string;
  album: string;
} | null> {
  try {
    const result = await osascript(
      'tell application "Music" to get {name, artist, album} of current track'
    );
    const [name, artist, album] = result.split(", ");
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

// ── Action handlers ───────────────────────────────────────────────

async function handlePlay(
  params: Record<string, string>
): Promise<SkillResponse> {
  const song = params.song;
  const artist = params.artist;

  if (song) {
    // Search and play a specific song
    const searchQuery = artist ? `${song} ${artist}` : song;
    try {
      // Search in the user's library first
      await osascript(
        `tell application "Music"
          activate
          set searchResults to search playlist "Library" for "${escapeAS(searchQuery)}"
          if (count of searchResults) > 0 then
            play item 1 of searchResults
          else
            play
          end if
        end tell`
      );
      const track = await getCurrentTrack();
      return {
        success: true,
        voice: track
          ? `Playing ${track.name} by ${track.artist}`
          : `Searching for ${searchQuery}`,
        sound: "Pop",
      };
    } catch (err) {
      // Fallback: just play
      await osascript('tell application "Music" to play');
      return {
        success: true,
        voice: `I couldn't find "${searchQuery}", but music is playing now.`,
      };
    }
  }

  // Generic play
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
  await osascript('tell application "Music" to pause');
  return {
    success: true,
    voice: "Music paused",
    sound: "Tink",
  };
}

async function handleResume(): Promise<SkillResponse> {
  await osascript('tell application "Music" to play');
  const track = await getCurrentTrack();
  return {
    success: true,
    voice: track ? `Resuming ${track.name}` : "Resuming playback",
    sound: "Pop",
  };
}

async function handleSkip(): Promise<SkillResponse> {
  await osascript('tell application "Music" to next track');
  // Small delay for track to change
  await new Promise((r) => setTimeout(r, 500));
  const track = await getCurrentTrack();
  return {
    success: true,
    voice: track ? `Now playing ${track.name} by ${track.artist}` : "Skipped to next track",
    sound: "Pop",
  };
}

async function handlePrevious(): Promise<SkillResponse> {
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
  const newVol = Math.min(100, (Number(current) || 50) + 15);
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
  const newVol = Math.max(0, (Number(current) || 50) - 15);
  await osascript(
    `tell application "Music" to set sound volume to ${newVol}`
  );
  return {
    success: true,
    voice: `Volume down to ${newVol} percent`,
    sound: "Blow",
  };
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

// ── Skill definition ──────────────────────────────────────────────

export const musicSkill: Skill = {
  name: "music",
  description:
    "Controls Apple Music playback. " +
    "Activate when the user gives a clear command to play, pause, skip, or adjust music. " +
    'Examples: "play some music", "skip this song", "turn it up", "pause the music". ' +
    "Do NOT activate when the user is merely talking ABOUT music.",

  actions: [
    {
      name: "play",
      description: "Start playing music, optionally a specific song or artist",
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
