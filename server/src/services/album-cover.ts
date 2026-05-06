// Server-side procedural album cover generation.
// Generates an SVG gradient + decorative shapes + song title overlay,
// rasterizes to PNG via @resvg/resvg-js (pure WASM — no native deps,
// Alpine-safe). CPU-only — does not touch V100.
//
// Output: 1024x1024 PNG buffer suitable for ID3 cover-art embedding.

import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import { Resvg } from '@resvg/resvg-js';

// Locate the bundled Brosnoirs font. Resolved at module load so we don't pay
// the lookup cost per render.
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Compiled location: /app/server/dist/services/album-cover.js
// Font is shipped at /app/server/branding/fonts/Brosnoirs.ttf (see Dockerfile)
const FONT_PATH = path.resolve(__dirname, '../../branding/fonts/Brosnoirs.ttf');
const FONT_AVAILABLE = fs.existsSync(FONT_PATH);
if (!FONT_AVAILABLE) {
  console.warn('[album-cover] Brosnoirs.ttf not found at', FONT_PATH, '— covers will render with system fallback (or empty if Alpine).');
}

// Curated music-themed palettes — each [bg, primary, secondary, accent]
// Picked deterministically by hashing the song title so the same song always
// gets the same cover.
const PALETTES: Array<[string, string, string, string]> = [
  ['#0F0F1E', '#FF006E', '#8338EC', '#FFBE0B'],  // Synthwave Magenta
  ['#1A1A2E', '#16213E', '#E94560', '#F5A623'],  // Sunset Vibes
  ['#0D1B2A', '#1B263B', '#778DA9', '#E0E1DD'],  // Ocean Depths
  ['#1F1F1F', '#FFD93D', '#FF6B6B', '#4ECDC4'],  // Reggae Gold
  ['#1A1A2E', '#0F3460', '#16C79A', '#FF9F1C'],  // Tropical Roots
  ['#000000', '#7209B7', '#F72585', '#4CC9F0'],  // Electric Dance
  ['#0E1116', '#21262D', '#58A6FF', '#F78166'],  // Tech Dark
  ['#1B1B1B', '#A4243B', '#D8973C', '#BD632F'],  // Earthy Roots
  ['#0A0A0A', '#1B1B3A', '#FFC857', '#FF6F61'],  // Carnival Night
  ['#13141C', '#23244D', '#7DC4FF', '#C7F8FF'],  // Cool Crystal
];

function hashString(str: string): number {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h) + str.charCodeAt(i);
    h = h & h;
  }
  return Math.abs(h) || 1;
}

/** Deterministic seeded PRNG so the same input always produces the same cover. */
class SeededRandom {
  private s: number;
  constructor(seed: number) { this.s = seed; }
  next(): number {
    this.s = (this.s * 1103515245 + 12345) & 0x7fffffff;
    return this.s / 0x7fffffff;
  }
  range(min: number, max: number): number { return min + this.next() * (max - min); }
}

function escapeXml(s: string): string {
  return s.replace(/[<>&'"]/g, c =>
    c === '<' ? '&lt;' : c === '>' ? '&gt;' : c === '&' ? '&amp;' : c === "'" ? '&apos;' : '&quot;'
  );
}

function pickStyleMotif(style: string | undefined): 'concentric' | 'bars' | 'waves' | 'grid' {
  // Music-style → visual motif mapping (rough heuristic, harmless if wrong)
  const s = (style || '').toLowerCase();
  if (s.match(/reggae|roots|dub|dancehall|afrobeat|caribbean/)) return 'concentric';
  if (s.match(/electronic|techno|edm|house|dnb/)) return 'bars';
  if (s.match(/r&b|soul|jazz|smooth|lounge/)) return 'waves';
  if (s.match(/hip[- ]?hop|rap|trap|drill|grime/)) return 'grid';
  // Default: concentric (works visually for most genres)
  return 'concentric';
}

function buildSvg(opts: {
  title: string;
  style?: string;
  seed: string;
  size: number;
}): string {
  const { title, style, seed, size } = opts;
  const rng = new SeededRandom(hashString(seed || title));
  const palette = PALETTES[Math.floor(rng.next() * PALETTES.length)];
  const [bg, c1, c2, c3] = palette;
  const motif = pickStyleMotif(style);

  // Title prep — trim long titles, tier the font sizing
  const safeTitle = (title || 'Untitled').slice(0, 32);
  const titleWords = safeTitle.split(/\s+/).filter(Boolean);
  // Layout title across max 3 lines
  const lines: string[] = [];
  if (titleWords.length <= 2) {
    lines.push(safeTitle.toUpperCase());
  } else if (titleWords.length <= 4) {
    const mid = Math.ceil(titleWords.length / 2);
    lines.push(titleWords.slice(0, mid).join(' ').toUpperCase());
    lines.push(titleWords.slice(mid).join(' ').toUpperCase());
  } else {
    const t = Math.ceil(titleWords.length / 3);
    lines.push(titleWords.slice(0, t).join(' ').toUpperCase());
    lines.push(titleWords.slice(t, t * 2).join(' ').toUpperCase());
    lines.push(titleWords.slice(t * 2).join(' ').toUpperCase());
  }
  // Font size: bigger = more impact (Owen wants BOLD)
  const longestLine = Math.max(...lines.map(l => l.length));
  const baseFontSize =
    longestLine <= 5 ? Math.floor(size * 0.32) :
    longestLine <= 9 ? Math.floor(size * 0.22) :
    longestLine <= 14 ? Math.floor(size * 0.15) :
    Math.floor(size * 0.10);

  // Decorative motifs (style-aware)
  let motifSvg = '';
  if (motif === 'concentric') {
    // Roots/reggae: concentric circles (sun-rays / dub plate vibe)
    const cx = rng.range(size * 0.2, size * 0.8);
    const cy = rng.range(size * 0.15, size * 0.4);
    for (let i = 0; i < 7; i++) {
      const r = (size / 14) * (i + 1);
      motifSvg += `<circle cx="${cx}" cy="${cy}" r="${r}" stroke="${c2}" stroke-opacity="${0.15 + (6 - i) * 0.05}" stroke-width="2" fill="none"/>`;
    }
  } else if (motif === 'bars') {
    // Electronic: vertical bars (equalizer)
    const numBars = 16;
    const w = size / numBars;
    for (let i = 0; i < numBars; i++) {
      const h = rng.range(size * 0.1, size * 0.4);
      const x = i * w;
      const y = size - h;
      motifSvg += `<rect x="${x}" y="${y}" width="${w * 0.8}" height="${h}" fill="${c2}" fill-opacity="${0.25 + rng.next() * 0.3}"/>`;
    }
  } else if (motif === 'waves') {
    // R&B/soul: smooth wave forms
    const yBase = size * 0.7;
    let path = `M 0 ${yBase} `;
    for (let x = 0; x <= size; x += size / 16) {
      const y = yBase + Math.sin(x * 0.02 + rng.next() * 6) * size * 0.05;
      path += `L ${x} ${y} `;
    }
    path += `L ${size} ${size} L 0 ${size} Z`;
    motifSvg += `<path d="${path}" fill="${c2}" fill-opacity="0.3"/>`;
  } else if (motif === 'grid') {
    // Hip-hop/rap: dot grid
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const cx = (c + 0.5) * (size / 8);
        const cy = (r + 0.5) * (size / 8);
        const radius = rng.range(size * 0.005, size * 0.025);
        motifSvg += `<circle cx="${cx}" cy="${cy}" r="${radius}" fill="${c2}" fill-opacity="${0.3 + rng.next() * 0.4}"/>`;
      }
    }
  }

  // Title text rendering — Brosnoirs (graffiti / dancehall display) with
  // dark stroke for high-contrast pop against any palette
  const titleY = size * 0.5;
  const lineGap = baseFontSize * 0.95;
  const titleSvg = lines.map((line, i) => {
    const y = titleY + (i - (lines.length - 1) / 2) * lineGap;
    return `<text x="${size / 2}" y="${y}" text-anchor="middle" dominant-baseline="middle"
              font-family="Brosnoirs, sans-serif"
              font-size="${baseFontSize}" fill="${c3}"
              stroke="${bg}" stroke-width="3" paint-order="stroke">${escapeXml(line)}</text>`;
  }).join('');

  // ── Infamous Media Productions clapboard logo (lower-right corner) ──
  // Reggae color scheme matches the original Wolfson Media Group logo Owen sent.
  // Placed in a 200×200 viewBox group, then scaled to 22% of cover side.
  const LOGO_VBOX = 200;
  const LOGO_SIZE = size * 0.22;
  const LOGO_X = size - LOGO_SIZE - size * 0.04;
  const LOGO_Y = size - LOGO_SIZE - size * 0.04;
  const REGGAE_GREEN = '#1F8B3B';
  const REGGAE_RED   = '#C8242C';
  const REGGAE_GOLD  = '#F2D71A';
  const logoSvg = `
    <g transform="translate(${LOGO_X} ${LOGO_Y}) scale(${LOGO_SIZE / LOGO_VBOX})">
      <!-- Slate body -->
      <rect x="10" y="50" width="180" height="140" fill="#0A0A0A" stroke="${REGGAE_GOLD}" stroke-width="2"/>
      <!-- Clapboard top piece (angled, hinged on left) -->
      <g transform="rotate(-6 25 35)">
        <rect x="10" y="22" width="180" height="32" fill="#0A0A0A"/>
        <polygon points="22,22 44,22 36,54 14,54" fill="#FFFFFF"/>
        <polygon points="64,22 86,22 78,54 56,54" fill="#FFFFFF"/>
        <polygon points="106,22 128,22 120,54 98,54" fill="#FFFFFF"/>
        <polygon points="148,22 170,22 162,54 140,54" fill="#FFFFFF"/>
        <circle cx="20" cy="30" r="3" fill="#FFFFFF"/>
        <circle cx="20" cy="44" r="3" fill="#FFFFFF"/>
      </g>
      <!-- Three text lines: INFAMOUS / MEDIA / PRODUCTIONS in reggae colors -->
      <text x="100" y="92" text-anchor="middle" font-family="Brosnoirs, sans-serif"
            font-size="38" fill="${REGGAE_GREEN}" stroke="black" stroke-width="1.5" paint-order="stroke">INFAMOUS</text>
      <text x="100" y="135" text-anchor="middle" font-family="Brosnoirs, sans-serif"
            font-size="42" fill="${REGGAE_RED}" stroke="black" stroke-width="1.5" paint-order="stroke">MEDIA</text>
      <text x="100" y="180" text-anchor="middle" font-family="Brosnoirs, sans-serif"
            font-size="34" fill="${REGGAE_GOLD}" stroke="black" stroke-width="1.5" paint-order="stroke">PRODUCTIONS</text>
    </g>
  `;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
    <defs>
      <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="${bg}"/>
        <stop offset="50%" stop-color="${c1}" stop-opacity="0.4"/>
        <stop offset="100%" stop-color="${bg}"/>
      </linearGradient>
      <radialGradient id="vignette" cx="0.5" cy="0.5" r="0.7">
        <stop offset="55%" stop-color="black" stop-opacity="0"/>
        <stop offset="100%" stop-color="black" stop-opacity="0.6"/>
      </radialGradient>
    </defs>
    <rect width="${size}" height="${size}" fill="url(#bg)"/>
    ${motifSvg}
    <rect width="${size}" height="${size}" fill="url(#vignette)"/>
    ${titleSvg}
    ${logoSvg}
  </svg>`;
}

/** Generate album cover PNG buffer from song metadata. */
export function generateAlbumCoverBuffer(opts: {
  title: string;
  style?: string;
  seed: string;
  size?: number;
}): Buffer {
  const size = opts.size || 1024;
  const svg = buildSvg({ title: opts.title, style: opts.style, seed: opts.seed, size });
  const resvg = new Resvg(svg, {
    background: '#000000',
    fitTo: { mode: 'width', value: size },
    font: FONT_AVAILABLE ? {
      fontFiles: [FONT_PATH],
      loadSystemFonts: false,
      defaultFontFamily: 'Brosnoirs',
    } : { loadSystemFonts: true },
  });
  return resvg.render().asPng();
}

/** Generate an album cover, save to public/audio/<userId>/<songId>-cover.png,
 *  return the /audio/... URL suitable for `cover_url` field. */
export async function ensureAlbumCover(opts: {
  userId: string;
  songId: string;
  title: string;
  style?: string;
  publicAudioDir: string;
  force?: boolean;
}): Promise<string> {
  const userDir = path.join(opts.publicAudioDir, opts.userId);
  await fs.promises.mkdir(userDir, { recursive: true });
  const filename = `${opts.songId}-cover.png`;
  const fullPath = path.join(userDir, filename);
  if (!opts.force && fs.existsSync(fullPath)) {
    return `/audio/${opts.userId}/${filename}`;
  }
  const buf = generateAlbumCoverBuffer({
    title: opts.title,
    style: opts.style,
    seed: opts.songId,
  });
  await fs.promises.writeFile(fullPath, buf);
  return `/audio/${opts.userId}/${filename}`;
}
