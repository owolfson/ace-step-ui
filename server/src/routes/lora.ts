import { Router, Response } from 'express';
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
  res.json(loraState);
});

// ── List endpoint (LoRA picker) ────────────────────────────────────────────
router.get('/list', authMiddleware, async (_req, res) => {
  try {
    const fs = await import('fs/promises');
    const path = await import('path');
    const root = process.env.LORA_ROOT || '/app/ACE-Step-1.5/checkpoints/loras';
    const out = [];
    async function walk(dir: string, collection: string) {
      let entries: any[] = [];
      try { entries = await fs.readdir(dir, { withFileTypes: true }); } catch { return; }
      let meta: any = {};
      for (const e of entries) {
        if (e.isFile() && e.name === 'meta.json') {
          try { meta = JSON.parse(await fs.readFile(path.join(dir, 'meta.json'), 'utf8')); } catch {}
        }
      }
      const def: any = meta.default || {};
      for (const e of entries) {
        if (e.isDirectory()) {
          await walk(path.join(dir, e.name), e.name);
        } else if (e.isFile() && e.name.endsWith('.safetensors')) {
          const full = path.join(dir, e.name);
          let sizeBytes = 0;
          try { sizeBytes = (await fs.stat(full)).size; } catch {}
          const m: any = meta[e.name] || {};
          const fallbackLabel = e.name
            .replace(/_adapter_model\.safetensors$/, '')
            .replace(/\.safetensors$/, '')
            .replace(/_/g, ' ')
            .replace(/\s+/g, ' ')
            .trim() || e.name;
          out.push({
            path: full,
            label: m.label || fallbackLabel,
            collection: def.collection || collection,
            description: m.description || def.description || '',
            sizeBytes,
          });
        }
      }
    }
    await walk(root, path.basename(root));
    out.sort((a, b) => (a.collection + a.label).localeCompare(b.collection + b.label));
    res.json({ loras: out, count: out.length });
  } catch (error) {
    console.error('[LoRA] List error:', error);
    res.status(500).json({ error: (error && error.message) || 'Failed to list LoRAs' });
  }
});

export default router;
