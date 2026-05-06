import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import NodeID3 from 'node-id3';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_AUDIO_DIR = path.resolve(__dirname, '../../public/audio');

export interface SongRecord {
  id: string;
  title?: string;
  lyrics?: string;
  style?: string;
  caption?: string;
  cover_url?: string;
  audio_url?: string;
  bpm?: number;
  duration?: number;
  key_scale?: string;
}

function audioUrlToPath(audioUrl: string | undefined | null): string | null {
  if (!audioUrl) return null;
  if (audioUrl.startsWith('/audio/')) {
    return path.join(PUBLIC_AUDIO_DIR, audioUrl.replace('/audio/', ''));
  }
  return null; // remote URL — cannot tag in place
}

function extractGenre(caption: string | undefined): string {
  if (!caption) return 'Music';
  const tags = caption.split(',').map(t => t.trim()).filter(Boolean);
  if (!tags.length) return 'Music';
  // First tag is conventionally the genre
  return tags[0].split(/\s+/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

export async function tagMp3(song: SongRecord): Promise<boolean> {
  const filePath = audioUrlToPath(song.audio_url);
  if (!filePath) return false;
  if (!filePath.endsWith('.mp3')) return false;
  if (!fs.existsSync(filePath)) {
    console.warn('[mp3tag] file missing:', filePath);
    return false;
  }
  const tags: any = {
    title: song.title || 'Untitled',
    // Brand defaults — override per-deployment with MP3_TAG_ARTIST / MP3_TAG_ALBUM
    // env vars in docker-compose.yml.
    artist: process.env.MP3_TAG_ARTIST || 'Cubane Studio',
    album: process.env.MP3_TAG_ALBUM || 'Infamous Media Productions',
    year: String(new Date().getFullYear()),
    genre: extractGenre(song.caption || song.style),
    comment: { language: 'eng', text: song.caption || song.style || '' },
    userDefinedText: [
      { description: 'STYLE', value: song.style || '' },
      { description: 'CAPTION', value: song.caption || '' },
      // Provenance: AI-generated music should be transparent in metadata
      { description: 'GENERATED_BY', value: 'ACE-Step v1.5 + Cubane Studio' },
    ],
  };
  if (song.bpm && song.bpm > 0) tags.bpm = String(Math.round(song.bpm));
  if (song.key_scale) tags.initialKey = song.key_scale;
  const coverPath = audioUrlToPath(song.cover_url);
  if (coverPath && fs.existsSync(coverPath)) {
    try {
      const buf = fs.readFileSync(coverPath);
      const ext = path.extname(coverPath).toLowerCase();
      tags.image = {
        mime: ext === '.png' ? 'image/png' : 'image/jpeg',
        type: { id: 3, name: 'front cover' },
        description: song.title || 'Album cover',
        imageBuffer: buf,
      };
    } catch (e: any) {
      console.warn('[mp3tag] cover read failed:', e.message);
    }
  }
  try {
    const result = NodeID3.write(tags, filePath);
    if (result === true) {
      console.log('[mp3tag] ✓', path.basename(filePath), '|', tags.title, '|', tags.genre);
      return true;
    }
    console.warn('[mp3tag] write non-true:', result);
    return false;
  } catch (e: any) {
    console.error('[mp3tag] write error:', e.message);
    return false;
  }
}
