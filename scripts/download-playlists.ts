#!/usr/bin/env bun
/**
 * Download mood playlists from YouTube using yt-dlp.
 *
 * Usage:
 *   bun run scripts/download-playlists.ts                    # download all
 *   bun run scripts/download-playlists.ts --quadrant uplift  # download one quadrant
 *   bun run scripts/download-playlists.ts --dry-run          # show what would be downloaded
 *
 * Prerequisites:
 *   brew install yt-dlp ffmpeg
 */

import { readFile, writeFile, mkdir, readdir } from "fs/promises";
import { resolve, join, basename } from "path";
import { parseArgs } from "util";
import { parse as parseYaml } from "yaml";

// ── Types ──────────────────────────────────────────────────────────

interface PlaylistSource {
  description: string;
  urls: string[];
  queries: string[];
}

interface TrackMeta {
  filename: string;
  title: string;
  artist: string;
  duration_seconds: number;
  bpm: number | null;
  source_url: string;
  downloaded_at: string;
  mood_tags: string[];
}

interface PlaylistMeta {
  quadrant: string;
  description: string;
  tracks: TrackMeta[];
}

// ── Paths ──────────────────────────────────────────────────────────

const DATA_DIR = resolve(import.meta.dir, "../data/playlists");
const CONFIG_PATH = join(DATA_DIR, "config.yaml");

// ── CLI ────────────────────────────────────────────────────────────

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    quadrant: { type: "string" },
    "dry-run": { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});

if (values.help) {
  console.log(`
  download-playlists — fetch mood playlist audio via yt-dlp

  Usage:
    bun run scripts/download-playlists.ts                    # download all quadrants
    bun run scripts/download-playlists.ts --quadrant uplift  # download one quadrant
    bun run scripts/download-playlists.ts --dry-run          # show what would happen

  Prerequisites:
    brew install yt-dlp ffmpeg
  `);
  process.exit(0);
}

// ── Helpers ────────────────────────────────────────────────────────

async function checkPrereqs(): Promise<void> {
  for (const cmd of ["yt-dlp", "ffmpeg"]) {
    const proc = Bun.spawn(["which", cmd], { stdout: "pipe", stderr: "pipe" });
    const exit = await proc.exited;
    if (exit !== 0) {
      console.error(`  ✗ ${cmd} not found. Install with: brew install ${cmd}`);
      process.exit(1);
    }
  }
}

async function loadConfig(): Promise<Record<string, PlaylistSource>> {
  const raw = await readFile(CONFIG_PATH, "utf-8");
  return parseYaml(raw) as Record<string, PlaylistSource>;
}

async function loadExistingMeta(dir: string): Promise<PlaylistMeta> {
  const metaPath = join(dir, "_meta.json");
  try {
    const raw = await readFile(metaPath, "utf-8");
    return JSON.parse(raw) as PlaylistMeta;
  } catch {
    return { quadrant: "", description: "", tracks: [] };
  }
}

/**
 * Read yt-dlp .info.json sidecar files and build track metadata.
 */
async function scanInfoFiles(dir: string): Promise<TrackMeta[]> {
  const files = await readdir(dir);
  const infoFiles = files.filter(f => f.endsWith(".info.json"));
  const tracks: TrackMeta[] = [];

  for (const infoFile of infoFiles) {
    try {
      const raw = await readFile(join(dir, infoFile), "utf-8");
      const info = JSON.parse(raw);

      // Find the corresponding audio file
      const baseName = infoFile.replace(".info.json", "");
      const audioFile = files.find(f =>
        f.startsWith(baseName) && (f.endsWith(".opus") || f.endsWith(".mp3") || f.endsWith(".m4a"))
      );

      if (!audioFile) continue;

      tracks.push({
        filename: audioFile,
        title: info.title || basename(audioFile, ".opus"),
        artist: info.uploader || info.channel || "Unknown",
        duration_seconds: Math.round(info.duration || 0),
        bpm: null, // BPM detection deferred to v2
        source_url: info.webpage_url || info.original_url || "",
        downloaded_at: new Date().toISOString(),
        mood_tags: [],
      });
    } catch (e) {
      console.warn(`  ⚠ failed to parse ${infoFile}: ${e}`);
    }
  }

  return tracks;
}

async function downloadQuadrant(
  quadrant: string,
  source: PlaylistSource,
  dryRun: boolean,
): Promise<void> {
  const dir = join(DATA_DIR, quadrant);
  await mkdir(dir, { recursive: true });

  console.log(`\n  ▸ ${quadrant}: ${source.description}`);

  // Load existing metadata to skip already-downloaded tracks
  const existing = await loadExistingMeta(dir);
  const existingUrls = new Set(existing.tracks.map(t => t.source_url));

  // Collect all URLs to download (explicit + search queries)
  const allUrls = [
    ...source.urls,
    ...source.queries,
  ];

  if (allUrls.length === 0) {
    console.log("    (no URLs or queries configured)");
    return;
  }

  for (const url of allUrls) {
    const isSearch = url.startsWith("ytsearch");

    if (dryRun) {
      console.log(`    [dry-run] would download: ${url}`);
      continue;
    }

    console.log(`    downloading: ${url}`);

    const args = [
      "yt-dlp",
      "--extract-audio",
      "--audio-format", "opus",
      "--audio-quality", "5",
      "--output", join(dir, "%(autonumber)03d-%(title)s.%(ext)s"),
      "--write-info-json",
      "--no-overwrites",
      "--ignore-errors",
      "--quiet",
      "--progress",
    ];

    // For explicit URLs, download one at a time
    // For search queries, let yt-dlp handle the search
    if (!isSearch) {
      args.push("--no-playlist");
    }

    args.push(url);

    const proc = Bun.spawn(args, {
      stdout: "inherit",
      stderr: "inherit",
    });

    const exitCode = await proc.exited;
    if (exitCode !== 0) {
      console.error(`    ✗ failed: ${url} (exit ${exitCode})`);
    }
  }

  // Rebuild _meta.json from downloaded files
  if (!dryRun) {
    const tracks = await scanInfoFiles(dir);
    const meta: PlaylistMeta = {
      quadrant,
      description: source.description,
      tracks,
    };
    await writeFile(
      join(dir, "_meta.json"),
      JSON.stringify(meta, null, 2) + "\n",
    );
    console.log(`    ✓ ${tracks.length} tracks indexed in _meta.json`);
  }
}

// ── Main ───────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log("  download-playlists");
  console.log(`  config: ${CONFIG_PATH}`);

  if (!values["dry-run"]) {
    await checkPrereqs();
  }

  const config = await loadConfig();
  const quadrants = values.quadrant
    ? [values.quadrant]
    : Object.keys(config);

  for (const q of quadrants) {
    const source = config[q];
    if (!source) {
      console.error(`  ✗ unknown quadrant: ${q}`);
      console.error(`  valid: ${Object.keys(config).join(", ")}`);
      process.exit(1);
    }
    await downloadQuadrant(q, source, values["dry-run"] ?? false);
  }

  console.log("\n  done.");
}

main().catch((err) => {
  console.error(`  ✗ ${err}`);
  process.exit(1);
});
