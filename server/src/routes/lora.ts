import { Router, Response } from 'express';
import { promises as fs } from 'fs';
import path from 'path';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.js';
import { getGradioClient } from '../services/gradio-client.js';

const router = Router();

// Local LoRA state tracking (Gradio doesn't have a dedicated status endpoint)
let loraState = {
  loaded: false,
  active: false,
  scale: 1.0,
  path: '',
};

// POST /api/lora/load — Load a LoRA adapter
router.post('/load', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { lora_path } = req.body;
    if (!lora_path || typeof lora_path !== 'string') {
      res.status(400).json({ error: 'lora_path is required' });
      return;
    }

    const client = await getGradioClient();
    const result = await client.predict('/load_lora', [lora_path]);
    const status = (result.data as unknown[])[0] as string;

    loraState = { loaded: true, active: true, scale: loraState.scale, path: lora_path };

    res.json({ message: status, lora_path, loaded: true });
  } catch (error) {
    console.error('[LoRA] Load error:', error);
    res.status(500).json({ error: error instanceof Error ? error.message : 'Failed to load LoRA' });
  }
});

// POST /api/lora/unload — Unload the current LoRA adapter
router.post('/unload', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const client = await getGradioClient();
    const result = await client.predict('/unload_lora', []);
    const status = (result.data as unknown[])[0] as string;

    loraState = { loaded: false, active: false, scale: 1.0, path: '' };

    res.json({ message: status });
  } catch (error) {
    console.error('[LoRA] Unload error:', error);
    res.status(500).json({ error: error instanceof Error ? error.message : 'Failed to unload LoRA' });
  }
});

// POST /api/lora/scale — Set LoRA scale (0.0 - 1.0)
router.post('/scale', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { scale } = req.body;
    if (typeof scale !== 'number' || scale < 0 || scale > 1) {
      res.status(400).json({ error: 'scale must be a number between 0 and 1' });
      return;
    }

    const client = await getGradioClient();
    const result = await client.predict('/set_lora_scale', [scale]);
    const status = (result.data as unknown[])[0] as string;

    loraState.scale = scale;

    res.json({ message: status, scale });
  } catch (error) {
    console.error('[LoRA] Scale error:', error);
    res.status(500).json({ error: error instanceof Error ? error.message : 'Failed to set LoRA scale' });
  }
});

// POST /api/lora/toggle — Toggle LoRA on/off
router.post('/toggle', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { enabled } = req.body;
    const useLoRA = typeof enabled === 'boolean' ? enabled : !loraState.active;

    const client = await getGradioClient();
    const result = await client.predict('/set_use_lora', [useLoRA]);
    const status = (result.data as unknown[])[0] as string;

    loraState.active = useLoRA;

    res.json({ message: status, active: useLoRA });
  } catch (error) {
    console.error('[LoRA] Toggle error:', error);
    res.status(500).json({ error: error instanceof Error ? error.message : 'Failed to toggle LoRA' });
  }
});

// GET /api/lora/status — Get current LoRA state
router.get('/status', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  // Query the native acestep-server for ground truth — local loraState can
  // drift if the server resets (e.g. container restart, /v1/init swap, or a
  // /v1/lora/load failure UI didn't observe). Server's /v1/lora/status is the
  // canonical source.
  try {
    const acestepUrl = (process.env.ACESTEP_API_URL || 'http://host.docker.internal:7861').replace(/\/+$/, '');
    const r = await fetch(acestepUrl + '/v1/lora/status', { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      const raw: any = await r.json();
      const d: any = raw.data || raw;
      // Adapt to UI shape (loraState) while exposing extras for multi-LoRA UI later
      const adapters: string[] = d.adapters || [];
      const activeAdapter = d.active_adapter || (adapters.length > 0 ? adapters[0] : '');
      const merged = {
        loaded: !!d.lora_loaded,
        active: !!d.use_lora,
        scale: typeof d.lora_scale === 'number' ? d.lora_scale : 1.0,
        path: loraState.path,           // keep last-known UI-side path (server doesn't return path)
        adapters,                        // extras: list of all loaded adapters
        active_adapter: activeAdapter,
        scales: d.scales || {},
      };
      // Reconcile local cache so future writes start from truth
      loraState.loaded = merged.loaded;
      loraState.active = merged.active;
      loraState.scale = merged.scale;
      res.json(merged);
      return;
    }
  } catch (e: any) {
    console.warn('[lora/status] server query failed, falling back to local cache:', e?.message);
  }
  res.json(loraState);
});

// ── List endpoint (LoRA picker) ────────────────────────────────────────────

const SAFETENSORS_EXT = '.safetensors';

interface LoraEntry {
  path: string;
  label: string;
  collection: string;
  description: string;
  sizeBytes: number;
}

/** Collapse a directory/file token into a human-friendly label. */
function prettifyName(token: string): string {
  return String(token || '')
    .replace(/[_-]+/g, ' ')
    .replace(/ +/g, ' ')
    .trim();
}

/**
 * Strip the generic "…adapter_model.safetensors" stem, leaving only a
 * file-specific qualifier (e.g. "male_vocals", "voc_06_inst_14"). Returns an
 * empty string for a plain `adapter_model.safetensors`.
 */
function adapterVariant(fileName: string): string {
  const stem = fileName.slice(0, -SAFETENSORS_EXT.length);
  return prettifyName(stem.replace(/_*adapter_model$/, ''));
}

/**
 * Display label for one adapter. Names by the owning LoRA directory and only
 * appends the file variant when a single folder ships more than one adapter
 * weight (e.g. raspy-vocal-5-loras).
 */
function buildLoraLabel(loraName: string, variant: string, weightCount: number): string {
  const base = prettifyName(loraName);
  if (variant && variant.toLowerCase() !== base.toLowerCase()) {
    return weightCount > 1 ? `${base} — ${variant}` : (base || variant);
  }
  return base || variant;
}

// Internal entry carries the fields needed to dedupe, stripped before responding.
type CollectedLora = LoraEntry & { depth: number; key: string };

/**
 * Recursively collect `.safetensors` adapters under `root`. The first directory
 * level names the LoRA; deeper dirs (final/, adapter/, checkpoint-N/) inherit
 * that name so training shadow-copies don't surface as bare "adapter". Hidden
 * dirs (.cache, .git) are skipped.
 */
async function collectLoras(root: string): Promise<CollectedLora[]> {
  const found: CollectedLora[] = [];

  async function walk(dir: string, loraName: string | null): Promise<void> {
    let entries: any[] = [];
    try { entries = await fs.readdir(dir, { withFileTypes: true }); } catch { return; }

    let meta: any = {};
    for (const e of entries) {
      if (e.isFile() && e.name === 'meta.json') {
        try { meta = JSON.parse(await fs.readFile(path.join(dir, 'meta.json'), 'utf8')); } catch {}
      }
    }
    const def: any = meta.default || {};
    const weightCount = entries.filter(e => e.isFile() && e.name.endsWith(SAFETENSORS_EXT)).length;

    for (const e of entries) {
      if (e.name.startsWith('.')) continue;
      if (e.isDirectory()) {
        await walk(path.join(dir, e.name), loraName || e.name);
        continue;
      }
      if (!e.isFile() || !e.name.endsWith(SAFETENSORS_EXT)) continue;

      const full = path.join(dir, e.name);
      let sizeBytes = 0;
      try { sizeBytes = (await fs.stat(full)).size; } catch {}
      const m: any = meta[e.name] || {};
      const baseName = loraName || path.basename(dir);
      const variant = adapterVariant(e.name);

      found.push({
        path: full,
        label: m.label || buildLoraLabel(baseName, variant, weightCount),
        collection: def.collection || baseName,
        description: m.description || def.description || '',
        sizeBytes,
        depth: full.split('/').length,
        // Identity = LoRA dir + adapter variant. A top-level adapter and its
        // final/adapter/ shadow copy share this even when meta.json gives the
        // top-level a custom label, so they collapse to one entry.
        key: `${baseName} ${variant}`,
      });
    }
  }

  await walk(root, null);
  return found;
}

/** Collapse shadow copies that share a LoRA identity, keeping the shallowest path. */
function dedupeAdapters(entries: CollectedLora[]): LoraEntry[] {
  const byKey = new Map<string, CollectedLora>();
  for (const entry of entries) {
    const existing = byKey.get(entry.key);
    if (!existing || entry.depth < existing.depth) byKey.set(entry.key, entry);
  }
  return [...byKey.values()].map(({ depth, key, ...entry }) => entry);
}

router.get('/list', authMiddleware, async (_req, res) => {
  try {
    const root = process.env.LORA_ROOT || '/app/ACE-Step-1.5/checkpoints/loras';
    const loras = dedupeAdapters(await collectLoras(root))
      .sort((a, b) => (a.collection + a.label).localeCompare(b.collection + b.label));
    res.json({ loras, count: loras.length });
  } catch (error: any) {
    console.error('[LoRA] List error:', error);
    res.status(500).json({ error: (error && error.message) || 'Failed to list LoRAs' });
  }
});

export default router;
