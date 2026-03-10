/**
 * Responder — executes response actions when watchlist patterns trigger.
 *
 * Actions:
 *   - sound:        Play a macOS system sound (afplay)
 *   - voice:        Speak via ElevenLabs API (falls back to macOS `say`)
 *   - notification: Send a macOS notification (osascript)
 *
 * All actions are non-blocking and failure-tolerant.
 * The responder plays sound first, then voice, then notification —
 * creating a layered, gentle experience.
 */

import type { WatchlistMatch } from "./watchlist";

// ── ElevenLabs config (from env) ───────────────────────────────────

const ELEVEN_API_KEY = process.env.ELEVENLABS_API_KEY || "";
const ELEVEN_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || "";
const ELEVEN_API_URL = "https://api.elevenlabs.io/v1/text-to-speech";

// Voice settings tuned for gentle, soothing delivery
const ELEVEN_VOICE_SETTINGS = {
  stability: 0.7, // higher = more consistent, calmer
  similarity_boost: 0.75,
  style: 0.35, // subtle expressiveness
  use_speaker_boost: true,
};

const useElevenLabs = Boolean(ELEVEN_API_KEY && ELEVEN_VOICE_ID);

// ── Public API ─────────────────────────────────────────────────────

/**
 * Execute all configured responses for a watchlist match.
 * Order: sound → short pause → voice → notification.
 */
export async function respond(match: WatchlistMatch): Promise<void> {
  const { response } = match.pattern;
  const label = `[${match.pattern.category}/${match.pattern.id}]`;

  // 1. Play sound (non-blocking, very fast)
  if (response.sound) {
    await playSound(response.sound);
  }

  // 2. Small pause between sound and voice for a natural feel
  if (response.sound && response.voice) {
    await sleep(800);
  }

  // 3. Speak — try ElevenLabs first, fall back to macOS say
  if (response.voice) {
    const spoke = useElevenLabs
      ? await speakElevenLabs(response.voice)
      : false;

    if (!spoke) {
      await speakMacOS(response.voice, response.voiceName);
    }
  }

  // 4. Notification (appears after voice finishes)
  if (response.notification) {
    await sendNotification(`🎧 listen ${label}`, response.notification);
  }
}

// ── Sound ──────────────────────────────────────────────────────────

async function playSound(name: string): Promise<void> {
  const systemPath = `/System/Library/Sounds/${name}.aiff`;
  const soundPath = (await Bun.file(systemPath).exists())
    ? systemPath
    : name;

  try {
    const proc = Bun.spawn(["afplay", soundPath], {
      stdout: "ignore",
      stderr: "ignore",
    });
    await proc.exited;
  } catch {
    // non-critical
  }
}

// ── Voice: ElevenLabs ──────────────────────────────────────────────

/**
 * Speak text using ElevenLabs API.
 * Downloads MP3 to temp file and plays with afplay.
 * Returns true if successful, false to trigger fallback.
 */
async function speakElevenLabs(text: string): Promise<boolean> {
  const tmpPath = `/tmp/listen-voice-${Date.now()}.mp3`;

  try {
    const res = await fetch(
      `${ELEVEN_API_URL}/${ELEVEN_VOICE_ID}`,
      {
        method: "POST",
        headers: {
          "xi-api-key": ELEVEN_API_KEY,
          "Content-Type": "application/json",
          Accept: "audio/mpeg",
        },
        body: JSON.stringify({
          text,
          model_id: "eleven_multilingual_v2",
          voice_settings: ELEVEN_VOICE_SETTINGS,
        }),
      }
    );

    if (!res.ok) {
      console.error(
        `  ⚠ ElevenLabs ${res.status}: ${await res.text().catch(() => "unknown")}`
      );
      return false;
    }

    // Write audio to temp file
    const audioBuffer = await res.arrayBuffer();
    await Bun.write(tmpPath, audioBuffer);

    // Play it
    const proc = Bun.spawn(["afplay", tmpPath], {
      stdout: "ignore",
      stderr: "ignore",
    });
    await proc.exited;

    // Cleanup
    try {
      await Bun.file(tmpPath).unlink();
    } catch {}

    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`  ⚠ ElevenLabs failed: ${msg} (falling back to macOS say)`);

    // Cleanup on error
    try {
      await Bun.file(tmpPath).unlink();
    } catch {}

    return false;
  }
}

// ── Voice: macOS TTS (fallback) ────────────────────────────────────

async function speakMacOS(
  text: string,
  voiceName?: string
): Promise<void> {
  const voice = voiceName || "Samantha";
  const args = ["say", "-v", voice, "-r", "175", text];

  try {
    const proc = Bun.spawn(args, {
      stdout: "ignore",
      stderr: "ignore",
    });
    await proc.exited;
  } catch {
    // non-critical
  }
}

// ── Notification ───────────────────────────────────────────────────

async function sendNotification(
  title: string,
  body: string
): Promise<void> {
  const escapeAS = (s: string): string =>
    s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n");

  const script = `display notification "${escapeAS(body)}" with title "${escapeAS(title)}" sound name "default"`;

  try {
    const proc = Bun.spawn(["osascript", "-"], {
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });
    proc.stdin.write(script);
    proc.stdin.end();
    await proc.exited;
  } catch {
    // non-critical
  }
}

// ── Util ───────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
