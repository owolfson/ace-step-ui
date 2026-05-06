import { Router, Response } from 'express';
import { tagMp3 as _tagMp3 } from '../services/mp3tagging.js';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import { pool } from '../db/pool.js';
import { generateUUID } from '../db/sqlite.js';
import { config } from '../config/index.js';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.js';
import { getGradioClient } from '../services/gradio-client.js';
import {
  generateMusicViaAPI,
  getJobStatus,
  getAudioStream,
  discoverEndpoints,
  checkSpaceHealth,
  cleanupJob,
  getJobRawResponse,
  downloadAudioToBuffer,
  resolvePythonPath,
} from '../services/acestep.js';
import { getStorageProvider } from '../services/storage/factory.js';

const router = Router();

// Auto-generate a song title from lyrics or style when none is provided
function autoTitle(params: { title?: string; lyrics?: string; instrumental?: boolean; style?: string; songDescription?: string }): string {
  if (params.title?.trim()) return params.title.trim();

  // Try first meaningful lyric line (skip section markers like [verse], [chorus])
  if (!params.instrumental && params.lyrics) {
    for (const line of params.lyrics.split('\n')) {
      const t = line.trim();
      if (t && !/^\[.*\]$/.test(t)) {
        return t.length > 40 ? t.slice(0, 40).trimEnd() + '…' : t;
      }
    }
  }

  // Fall back to first 4 words of style or description
  const source = params.style || params.songDescription || '';
  if (source) {
    const words = source.trim().split(/\s+/).slice(0, 4).join(' ');
    return words.charAt(0).toUpperCase() + words.slice(1);
  }

  return 'Untitled';
}

const audioUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25MB max
  fileFilter: (_req, file, cb) => {
    const allowedTypes = [
      'audio/mpeg',
      'audio/mp3', // Alternative MIME type for MP3
      'audio/mpeg3',
      'audio/x-mpeg-3',
      'audio/wav',
      'audio/x-wav',
      'audio/flac',
      'audio/x-flac',
      'audio/mp4',
      'audio/x-m4a',
      'audio/aac',
      'audio/ogg',
      'audio/webm',
      'video/mp4',
    ];

    // Also check file extension as fallback
    const allowedExtensions = ['.mp3', '.wav', '.flac', '.m4a', '.mp4', '.aac', '.ogg', '.webm', '.opus'];
    const fileExt = file.originalname.toLowerCase().match(/\.[^.]+$/)?.[0];

    if (allowedTypes.includes(file.mimetype) || (fileExt && allowedExtensions.includes(fileExt))) {
      cb(null, true);
    } else {
      cb(new Error(`Invalid file type. Only common audio formats are allowed. Received: ${file.mimetype} (${file.originalname})`));
    }
  }
});

interface GenerateBody {
  // Mode
  customMode: boolean;

  // Simple Mode
  songDescription?: string;

  // Custom Mode
  lyrics: string;
  style: string;
  title: string;

  // Common
  instrumental: boolean;
  vocalLanguage?: string;

  // Music Parameters
  duration?: number;
  bpm?: number;
  keyScale?: string;
  timeSignature?: string;

  // Generation Settings
  inferenceSteps?: number;
  guidanceScale?: number;
  batchSize?: number;
  randomSeed?: boolean;
  seed?: number;
  thinking?: boolean;
  audioFormat?: 'mp3' | 'flac';
  inferMethod?: 'ode' | 'sde';
  shift?: number;

  // LM Parameters
  lmTemperature?: number;
  lmCfgScale?: number;
  lmTopK?: number;
  lmTopP?: number;
  lmNegativePrompt?: string;
  lmBackend?: 'pt' | 'vllm';
  lmModel?: string;

  // Expert Parameters
  referenceAudioUrl?: string;
  sourceAudioUrl?: string;
  referenceAudioTitle?: string;
  sourceAudioTitle?: string;
  audioCodes?: string;
  repaintingStart?: number;
  repaintingEnd?: number;
  instruction?: string;
  audioCoverStrength?: number;
  taskType?: string;
  useAdg?: boolean;
  cfgIntervalStart?: number;
  cfgIntervalEnd?: number;
  customTimesteps?: string;
  useCotMetas?: boolean;
  useCotCaption?: boolean;
  useCotLanguage?: boolean;
  autogen?: boolean;
  constrainedDecodingDebug?: boolean;
  allowLmBatch?: boolean;
  getScores?: boolean;
  getLrc?: boolean;
  scoreScale?: number;
  lmBatchChunkSize?: number;
  trackName?: string;
  completeTrackClasses?: string[];
  isFormatCaption?: boolean;

  // Model selection
  ditModel?: string;
}

router.post('/upload-audio', authMiddleware, (req: AuthenticatedRequest, res: Response, next: Function) => {
  audioUpload.single('audio')(req, res, (err: any) => {
    if (err) {
      res.status(400).json({ error: err.message || 'Invalid file upload' });
      return;
    }
    next();
  });
}, async (req: AuthenticatedRequest, res: Response) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'Audio file is required' });
      return;
    }

    const storage = getStorageProvider();
    const extFromName = path.extname(req.file.originalname || '').toLowerCase();
    const extFromType = (() => {
      switch (req.file.mimetype) {
        case 'audio/mpeg':
          return '.mp3';
        case 'audio/wav':
        case 'audio/x-wav':
          return '.wav';
        case 'audio/flac':
        case 'audio/x-flac':
          return '.flac';
        case 'audio/ogg':
          return '.ogg';
        case 'audio/mp4':
        case 'audio/x-m4a':
        case 'audio/aac':
          return '.m4a';
        case 'audio/webm':
          return '.webm';
        case 'video/mp4':
          return '.mp4';
        default:
          return '';
      }
    })();
    const ext = extFromName || extFromType || '.audio';
    const key = `references/${req.user!.id}/${Date.now()}-${generateUUID()}${ext}`;
    const storedKey = await storage.upload(key, req.file.buffer, req.file.mimetype);
    const publicUrl = storage.getPublicUrl(storedKey);

    res.json({ url: publicUrl, key: storedKey });
  } catch (error) {
    console.error('Upload reference audio error:', error);
    res.status(500).json({ error: 'Failed to upload audio' });
  }
});

router.post('/', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const {
      customMode,
      songDescription,
      lyrics,
      style,
      title,
      instrumental,
      vocalLanguage,
      duration,
      bpm,
      keyScale,
      timeSignature,
      inferenceSteps,
      guidanceScale,
      batchSize,
      randomSeed,
      seed,
      thinking,
      audioFormat,
      inferMethod,
      shift,
      lmTemperature,
      lmCfgScale,
      lmTopK,
      lmTopP,
      lmNegativePrompt,
      lmBackend,
      lmModel,
      referenceAudioUrl,
      sourceAudioUrl,
      referenceAudioTitle,
      sourceAudioTitle,
      audioCodes,
      repaintingStart,
      repaintingEnd,
      instruction,
      audioCoverStrength,
      taskType,
      useAdg,
      cfgIntervalStart,
      cfgIntervalEnd,
      customTimesteps,
      useCotMetas,
      useCotCaption,
      useCotLanguage,
      autogen,
      constrainedDecodingDebug,
      allowLmBatch,
      getScores,
      getLrc,
      scoreScale,
      lmBatchChunkSize,
      trackName,
      completeTrackClasses,
      isFormatCaption,
      ditModel,
    } = req.body as GenerateBody;

    if (!customMode && !songDescription) {
      res.status(400).json({ error: 'Song description required for simple mode' });
      return;
    }

    if (customMode && !style && !lyrics && !referenceAudioUrl) {
      res.status(400).json({ error: 'Style, lyrics, or reference audio required for custom mode' });
      return;
    }

    const params = {
      customMode,
      songDescription,
      lyrics,
      style,
      title,
      instrumental,
      vocalLanguage,
      duration,
      bpm,
      keyScale,
      timeSignature,
      inferenceSteps,
      guidanceScale,
      batchSize,
      randomSeed,
      seed,
      thinking,
      audioFormat,
      inferMethod,
      shift,
      lmTemperature,
      lmCfgScale,
      lmTopK,
      lmTopP,
      lmNegativePrompt,
      lmBackend,
      lmModel,
      referenceAudioUrl,
      sourceAudioUrl,
      referenceAudioTitle,
      sourceAudioTitle,
      audioCodes,
      repaintingStart,
      repaintingEnd,
      instruction,
      audioCoverStrength,
      taskType,
      useAdg,
      cfgIntervalStart,
      cfgIntervalEnd,
      customTimesteps,
      useCotMetas,
      useCotCaption,
      useCotLanguage,
      autogen,
      constrainedDecodingDebug,
      allowLmBatch,
      getScores,
      getLrc,
      scoreScale,
      lmBatchChunkSize,
      trackName,
      completeTrackClasses,
      isFormatCaption,
      ditModel,
    };

    // Create job record in database
    const localJobId = generateUUID();
    await pool.query(
      `INSERT INTO generation_jobs (id, user_id, status, params, created_at, updated_at)
       VALUES (?, ?, 'queued', ?, datetime('now'), datetime('now'))`,
      [localJobId, req.user!.id, JSON.stringify(params)]
    );

    // Start generation
    const { jobId: hfJobId } = await generateMusicViaAPI(params);

    // Update job with ACE-Step task ID
    await pool.query(
      `UPDATE generation_jobs SET acestep_task_id = ?, status = 'running', updated_at = datetime('now') WHERE id = ?`,
      [hfJobId, localJobId]
    );

    res.json({
      jobId: localJobId,
      status: 'queued',
      queuePosition: 1,
    });
  } catch (error) {
    console.error('Generate error:', error);
    res.status(500).json({ error: (error as Error).message || 'Generation failed' });
  }
});

router.get('/status/:jobId', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const jobResult = await pool.query(
      `SELECT id, user_id, acestep_task_id, status, params, result, error, created_at
       FROM generation_jobs
       WHERE id = ?`,
      [req.params.jobId]
    );

    if (jobResult.rows.length === 0) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    const job = jobResult.rows[0];

    if (job.user_id !== req.user!.id) {
      res.status(403).json({ error: 'Access denied' });
      return;
    }

    // If job is still running, check ACE-Step status
    if (['pending', 'queued', 'running'].includes(job.status) && job.acestep_task_id) {
      try {
        const aceStatus = await getJobStatus(job.acestep_task_id);

        if (aceStatus.status !== job.status) {
          // Use optimistic lock: only update if status hasn't changed (prevents duplicate song creation)
          let updateQuery = `UPDATE generation_jobs SET status = ?, updated_at = datetime('now')`;
          const updateParams: unknown[] = [aceStatus.status];

          if (aceStatus.status === 'succeeded' && aceStatus.result) {
            updateQuery += `, result = ?`;
            updateParams.push(JSON.stringify(aceStatus.result));
          } else if (aceStatus.status === 'failed' && aceStatus.error) {
            updateQuery += `, error = ?`;
            updateParams.push(aceStatus.error);
          }

          updateQuery += ` WHERE id = ? AND status = ?`;
          updateParams.push(req.params.jobId, job.status);

          const updateResult = await pool.query(updateQuery, updateParams);
          const wasUpdated = updateResult.rowCount > 0;

          // If succeeded AND we were the first to update (optimistic lock), create song records
          if (aceStatus.status === 'succeeded' && aceStatus.result && wasUpdated) {
            const params = typeof job.params === 'string' ? JSON.parse(job.params) : job.params;
            const audioUrls = aceStatus.result.audioUrls.filter((url: string) => {
              const lower = url.toLowerCase();
              return lower.endsWith('.mp3') || lower.endsWith('.flac') || lower.endsWith('.wav');
            });
            const localPaths: string[] = [];
            const storage = getStorageProvider();

            for (let i = 0; i < audioUrls.length; i++) {
              const audioUrl = audioUrls[i];
              const variationSuffix = audioUrls.length > 1 ? ` (v${i + 1})` : '';
              const songTitle = autoTitle(params) + variationSuffix;

              const songId = generateUUID();

              try {
                const { buffer } = await downloadAudioToBuffer(audioUrl);
                const ext = audioUrl.includes('.flac') ? '.flac' : '.mp3';
                const storageKey = `${req.user!.id}/${songId}${ext}`;
                await storage.upload(storageKey, buffer, `audio/${ext.slice(1)}`);
                const storedPath = storage.getPublicUrl(storageKey);

                await pool.query(
                  `INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                                      duration, bpm, key_scale, time_signature, tags, is_public, generation_params,
                                      created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, datetime('now'), datetime('now'))`,
                  [
                    songId,
                    req.user!.id,
                    songTitle,
                    params.instrumental ? '[Instrumental]' : params.lyrics,
                    params.style,
                    params.style,
                    storedPath,
                    aceStatus.result.duration && aceStatus.result.duration > 0 ? aceStatus.result.duration : (params.duration && params.duration > 0 ? params.duration : 0),
                    aceStatus.result.bpm || params.bpm,
                    aceStatus.result.keyScale || params.keyScale,
                    aceStatus.result.timeSignature || params.timeSignature,
                    JSON.stringify([]),
                    JSON.stringify(params),
                  ]
                );

                localPaths.push(storedPath);
                                // ALBUM-COVER + MP3-TAG-HOOK-GENERATE: generate procedural cover then tag.
                                // Sequenced so tagMp3 picks up the freshly-saved cover_url.
                                (async () => {
                                  let coverUrl: string | undefined;
                                  try {
                                    const { ensureAlbumCover } = await import('../services/album-cover.js');
                                    const path = await import('path');
                                    const { fileURLToPath } = await import('url');
                                    const _fname = fileURLToPath(import.meta.url);
                                    const _dir = path.dirname(_fname);
                                    const PUBLIC_AUDIO_DIR = path.resolve(_dir, '../../public/audio');
                                    coverUrl = await ensureAlbumCover({
                                      userId: req.user!.id,
                                      songId,
                                      title: songTitle,
                                      style: params.style,
                                      publicAudioDir: PUBLIC_AUDIO_DIR,
                                    });
                                    // Persist cover_url to DB
                                    await pool.query('UPDATE songs SET cover_url = ? WHERE id = ?', [coverUrl, songId]);
                                  } catch (e: any) {
                                    console.warn('[album-cover]', e?.message);
                                  }
                                  await _tagMp3({
                                    id: songId,
                                    title: songTitle,
                                    lyrics: params.instrumental ? '[Instrumental]' : params.lyrics,
                                    style: params.style,
                                    caption: params.style,
                                    audio_url: storedPath,
                                    cover_url: coverUrl,
                                    bpm: aceStatus.result.bpm || params.bpm,
                                    key_scale: aceStatus.result.keyScale || params.keyScale,
                                    duration: aceStatus.result.duration || params.duration,
                                  }).catch((e: any) => console.warn('[mp3tag]', e.message));
                                })().catch((e: any) => console.warn('[post-gen]', e?.message));
              } catch (downloadError) {
                console.error(`Failed to download audio ${i + 1}:`, downloadError);
                // Still create song record with remote URL
                await pool.query(
                  `INSERT INTO songs (id, user_id, title, lyrics, style, caption, audio_url,
                                      duration, bpm, key_scale, time_signature, tags, is_public, generation_params,
                                      created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, datetime('now'), datetime('now'))`,
                  [
                    songId,
                    req.user!.id,
                    songTitle,
                    params.instrumental ? '[Instrumental]' : params.lyrics,
                    params.style,
                    params.style,
                    audioUrl,
                    aceStatus.result.duration && aceStatus.result.duration > 0 ? aceStatus.result.duration : (params.duration && params.duration > 0 ? params.duration : 0),
                    aceStatus.result.bpm || params.bpm,
                    aceStatus.result.keyScale || params.keyScale,
                    aceStatus.result.timeSignature || params.timeSignature,
                    JSON.stringify([]),
                    JSON.stringify(params),
                  ]
                );
                localPaths.push(audioUrl);
              }
            }

            aceStatus.result.audioUrls = localPaths;
            cleanupJob(job.acestep_task_id);
          }
        }

        res.json({
          jobId: req.params.jobId,
          status: aceStatus.status,
          queuePosition: aceStatus.queuePosition,
          etaSeconds: aceStatus.etaSeconds,
          progress: aceStatus.progress,
          stage: aceStatus.stage,
          result: aceStatus.result,
          error: aceStatus.error,
        });
        return;
      } catch (aceError) {
        console.error('ACE-Step status check error:', aceError);
      }
    }

    // Return stored status
    res.json({
      jobId: req.params.jobId,
      status: job.status,
      progress: undefined,
      stage: undefined,
      result: job.result && typeof job.result === 'string' ? JSON.parse(job.result) : job.result,
      error: job.error,
    });
  } catch (error) {
    console.error('Status check error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Audio proxy endpoint
router.get('/audio', async (req, res: Response) => {
  try {
    const audioPath = req.query.path as string;
    if (!audioPath) {
      res.status(400).json({ error: 'Path required' });
      return;
    }

    const audioResponse = await getAudioStream(audioPath);

    if (!audioResponse.ok) {
      res.status(audioResponse.status).json({ error: 'Failed to fetch audio' });
      return;
    }

    const contentType = audioResponse.headers.get('content-type');
    if (contentType) {
      res.setHeader('Content-Type', contentType);
    }

    const contentLength = audioResponse.headers.get('content-length');
    if (contentLength) {
      res.setHeader('Content-Length', contentLength);
    }

    const reader = audioResponse.body?.getReader();
    if (!reader) {
      res.status(500).json({ error: 'Failed to read audio stream' });
      return;
    }

    const pump = async (): Promise<void> => {
      const { done, value } = await reader.read();
      if (done) {
        res.end();
        return;
      }
      res.write(value);
      return pump();
    };

    await pump();
  } catch (error) {
    console.error('Audio proxy error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

router.get('/history', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await pool.query(
      `SELECT id, acestep_task_id, status, params, result, error, created_at
       FROM generation_jobs
       WHERE user_id = ?
       ORDER BY created_at DESC
       LIMIT 50`,
      [req.user!.id]
    );

    res.json({ jobs: result.rows });
  } catch (error) {
    console.error('Get history error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

router.get('/endpoints', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const endpoints = await discoverEndpoints();
    res.json({ endpoints });
  } catch (error) {
    console.error('Discover endpoints error:', error);
    res.status(500).json({ error: 'Failed to discover endpoints' });
  }
});

router.get('/models', async (_req, res: Response) => {
  try {
    const ACESTEP_DIR = process.env.ACESTEP_PATH || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../ACE-Step-1.5');
    const checkpointsDir = path.join(ACESTEP_DIR, 'checkpoints');

    // All known DiT models from Gradio's model_downloader.py registry:
    // - MAIN_MODEL_COMPONENTS includes "acestep-v15-turbo" (bundled with main download)
    // - SUBMODEL_REGISTRY includes the rest (separate HuggingFace repos, auto-downloaded on init)
    const ALL_DIT_MODELS = [
      'acestep-v15-turbo',             // default, from main model repo
      'acestep-v15-base',              // 3.5B atmospheric/calm instrumental
      'acestep-v15-sft',               // 3.5B aggressive/dense vocal-led
      'acestep-v15-xl-base',           // 5B cinematic/atmospheric (Cubane Dark Tech)
      'acestep-v15-xl-sft',            // 5B aggressive/dense XL
      'acestep-v15-xl-turbo',          // 5B fast iteration
      'acestep-v15-turbo-rl',          // turbo with RL refinement
      'acestep-v15-turbo-shift1',      // submodel
      'acestep-v15-turbo-shift3',      // submodel
      'acestep-v15-turbo-continuous',   // submodel
    ];

    // Query acestep-server's native /v1/model_inventory for the FULL list of
      // models on disk + their load status. Replaces the previous env-var-only
      // early-return (which was a workaround for spaceinvaderone's broken
      // /v1/models endpoint). After 2026-05-04 migration to acestep-server,
      // /v1/model_inventory returns proper {data:{models:[{name,is_loaded,...}],...}}.
      const _acestepUrl = (process.env.ACESTEP_API_URL || 'http://host.docker.internal:7861').replace(/\/+$/, '');
      try {
        const _r = await fetch(_acestepUrl + '/v1/model_inventory', { signal: AbortSignal.timeout(5000) });
        if (_r.ok) {
          const _raw: any = await _r.json();
          const _inv: any = _raw.data || _raw;
          const _models = (_inv.models || []).map((m: any) => ({
            name: m.name,
            is_active: m.is_loaded === true,
            is_preloaded: m.is_loaded === true,
            is_default: m.is_default === true,
            supported_task_types: m.supported_task_types || [],
          }));
          res.json({ models: _models, default_model: _inv.default_model });
          return;
        }
      } catch (_e: any) {
        console.warn('[models] /v1/model_inventory failed, falling back to env-var:', _e?.message);
      }
      // Fallback: ACESTEP_ACTIVE_MODEL env var if the live query failed
      const envActive = (process.env.ACESTEP_ACTIVE_MODEL || '').trim();
      if (envActive) {
        const activeOnly = [{ name: envActive, is_active: true, is_preloaded: true }];
        res.json({ models: activeOnly });
        return;
      }
      // Final fallback: continue to original /v1/models query path below
      // (kept for legacy compat — should not be reached after acestep-server is up)
      // Query Gradio /v1/models to get the currently loaded/active model
    let activeModel: string | null = null;
    try {
      const apiRes = await fetch(`${config.acestep.apiUrl}/v1/models`);
      if (apiRes.ok) {
        const data = await apiRes.json() as any;
        const gradioModels = data?.data?.models || data?.models || [];
        if (gradioModels.length > 0) {
          activeModel = gradioModels[0]?.name || null;
        }
      }
    } catch {
      // Gradio API unavailable
    }

    // Check which models are downloaded (exist on disk)
    // Matches Gradio's handler.py check_model_exists() and get_available_acestep_v15_models()
    const { existsSync, statSync } = await import('fs');
    const downloaded = new Set<string>();
    for (const model of ALL_DIT_MODELS) {
      const modelPath = path.join(checkpointsDir, model);
      try {
        if (existsSync(modelPath) && statSync(modelPath).isDirectory()) {
          downloaded.add(model);
        }
      } catch { /* skip */ }
    }

    // Also scan for any additional acestep-v15-* models on disk not in the registry
    // (e.g. user-trained or community models)
    try {
      const { readdirSync } = await import('fs');
      for (const entry of readdirSync(checkpointsDir)) {
        if (entry.startsWith('acestep-v15-') && statSync(path.join(checkpointsDir, entry)).isDirectory()) {
          downloaded.add(entry);
          if (!ALL_DIT_MODELS.includes(entry)) {
            ALL_DIT_MODELS.push(entry);
          }
        }
      }
    } catch { /* checkpoints dir may not exist */ }

    const models = ALL_DIT_MODELS.map(name => ({
      name,
      is_active: name === activeModel,
      is_preloaded: downloaded.has(name),
    }));

    // Sort: active first, then downloaded, then alphabetical
    models.sort((a, b) => {
      if (a.is_active !== b.is_active) return a.is_active ? -1 : 1;
      if (a.is_preloaded !== b.is_preloaded) return a.is_preloaded ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

    res.json({ models });
  } catch (error) {
    console.error('Models error:', error);
    res.status(500).json({ error: (error as Error).message });
  }
});

// GET /api/generate/random-description — Load a random simple description from Gradio
router.get('/random-description', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const client = await getGradioClient();
    const result = await client.predict('/load_random_simple_description', []);
    const data = result.data as unknown[];
    // Returns [description, instrumental, vocal_language]
    res.json({
      description: data[0] || '',
      instrumental: data[1] || false,
      vocalLanguage: data[2] || 'unknown',
    });
  } catch (error) {
    console.error('Random description error:', error);
    res.status(500).json({ error: (error as Error).message });
  }
});

router.get('/health', async (_req, res: Response) => {
  try {
    const healthy = await checkSpaceHealth();
    res.json({ healthy, aceStepUrl: config.acestep.apiUrl });
  } catch (error) {
    res.json({ healthy: false, aceStepUrl: config.acestep.apiUrl, error: (error as Error).message });
  }
});

router.get('/limits', async (_req, res: Response) => {
  try {
    const { spawn } = await import('child_process');
    const ACESTEP_DIR = process.env.ACESTEP_PATH || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../ACE-Step-1.5');
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const SCRIPTS_DIR = path.join(__dirname, '../../scripts');
    const LIMITS_SCRIPT = path.join(SCRIPTS_DIR, 'get_limits.py');
    const pythonPath = resolvePythonPath(ACESTEP_DIR);

    const result = await new Promise<{ success: boolean; data?: any; error?: string }>((resolve) => {
      const proc = spawn(pythonPath, [LIMITS_SCRIPT], {
        cwd: ACESTEP_DIR,
        env: {
          ...process.env,
          ACESTEP_PATH: ACESTEP_DIR,
        },
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => { stdout += data.toString(); });
      proc.stderr.on('data', (data) => { stderr += data.toString(); });

      proc.on('close', (code) => {
        if (code === 0 && stdout) {
          try {
            const parsed = JSON.parse(stdout);
            resolve({ success: true, data: parsed });
          } catch {
            resolve({ success: false, error: 'Failed to parse limits result' });
          }
        } else {
          resolve({ success: false, error: stderr || 'Failed to read limits' });
        }
      });

      proc.on('error', (err) => {
        resolve({ success: false, error: err.message });
      });
    });

    if (result.success && result.data) {
      res.json(result.data);
    } else {
      res.status(500).json({ error: result.error || 'Failed to load limits' });
    }
  } catch (error) {
    console.error('Limits error:', error);
    res.status(500).json({ error: (error as Error).message });
  }
});

router.get('/debug/:taskId', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const rawResponse = getJobRawResponse(req.params.taskId);
    if (!rawResponse) {
      res.status(404).json({ error: 'Job not found or no raw response available' });
      return;
    }
    res.json({ rawResponse });
  } catch (error) {
    res.status(500).json({ error: (error as Error).message });
  }
});

// Format endpoint - uses LLM to enhance style/lyrics
router.post('/format', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { caption, lyrics, bpm, duration, keyScale, timeSignature, temperature, topK, topP, lmModel, lmBackend } = req.body;

    if (!caption) {
      res.status(400).json({ error: 'Caption/style is required' });
      return;
    }

    const ACESTEP_API_URL = config.acestep.apiUrl;

    // Build param_obj for the REST API
    const paramObj: Record<string, unknown> = {};
    if (bpm && bpm > 0) paramObj.bpm = bpm;
    if (duration && duration > 0) paramObj.duration = duration;
    if (keyScale) paramObj.key = keyScale;
    if (timeSignature) paramObj.time_signature = timeSignature;

    // Primary path: call ACE-Step's /format_input REST endpoint (avoids Python spawn ENOENT on Windows)
    try {
      console.log(`[Format] Calling REST API: ${ACESTEP_API_URL}/format_input`);
      const apiRes = await fetch(`${ACESTEP_API_URL}/format_input`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: caption,
          lyrics: lyrics || '',
          temperature: temperature ?? 0.85,
          param_obj: paramObj,
        }),
        signal: AbortSignal.timeout(300_000), // 5 min — LLM may need to init first
      });

      const apiData = await apiRes.json() as any;

      if (!apiRes.ok || apiData.code !== 200) {
        const errMsg = apiData.error || apiData.detail || `Format API returned ${apiRes.status}`;
        console.error('[Format] API error:', errMsg);
        res.status(500).json({ success: false, error: errMsg });
        return;
      }

      const d = apiData.data;
      // ── Style-then-lyrics enhancement pipeline (Gemma3/Gemma4 via Ollama) ────
      // Step 0: ensure Ollama reachable (auto-start container if down).
      // Step 1: rewrite caption to ACE-Step tag style (always).
      // Step 2: re-detect vocal/instrumental intent on the ENHANCED caption.
      // Step 3: lyrics derived from ENHANCED caption — full 3-min song or [instrumental].
      // VRAM lifecycle: every Ollama call uses keep_alive:0 so the model unloads
      // from V100 within ~3s of completion, leaving room for ACE-Step.
      const _vKW = /\b(vocal|singing|sing|voice|singer|lyric|male|female|baritone|tenor|soprano|rap|rapper|mc|chant|choir|harmony|accent|refrain|verse|chorus)\b/i;
      const _iKW = /\b(instrumental|no\s*vocals?|no\s*lyrics?|no\s*singing|orchestral|cinematic\s+score|film\s+score|ambient)\b/i;
      const _origCaption = (req.body.caption as string) || '';
      const _userLyrics = (req.body.lyrics as string || '').trim();
      // Dynamic lyric line budget computed from the user's requested duration.
      // ACE-Step phrases lines into the render window — too many = rushed vocals,
      // too few = sparse. Heuristic: ~3.5s per sung line average. If duration
      // is missing or zero, default to 180s (3 min) since LLM still needs SOME target.
      const _reqDur = Number(req.body.duration) > 0 ? Number(req.body.duration) : 180;
      const _durSec = Math.round(_reqDur);
      const _durMinStr = (_durSec / 60).toFixed(1);
      const _maxLines = Math.max(Math.round(_durSec / 3.5), 12);
      const _minLines = Math.max(Math.round(_maxLines * 0.65), 8);
      const _hardCap = _maxLines + 5;
      const _ollamaBase = (process.env.OLLAMA_URL || 'http://host.docker.internal:11434').replace(/\/+$/, '');
      const _ollamaUrl = _ollamaBase + '/api/generate';
      const _model = process.env.LYRICS_MODEL || 'gemma3:12b';

      // ── Step 0: ensure Ollama reachable, auto-start if needed ────────────
      async function _ollamaReady(timeoutMs: number): Promise<boolean> {
        try {
          const r = await fetch(_ollamaBase + '/api/tags', { signal: AbortSignal.timeout(Math.min(timeoutMs, 3000)) });
          return r.ok;
        } catch { return false; }
      }
      async function _dockerStart(name: string): Promise<boolean> {
        try {
          const _http = await import('http');
          await new Promise<void>((resolve, reject) => {
            const r = _http.request({ socketPath: '/var/run/docker.sock', method: 'POST', path: `/containers/${name}/start`, timeout: 10000 }, (res: any) => {
              res.resume();
              res.on('end', () => (res.statusCode! < 400 || res.statusCode === 304) ? resolve() : reject(new Error('docker start status ' + res.statusCode)));
            });
            r.on('error', reject);
            r.end();
          });
          return true;
        } catch (e: any) {
          console.warn('[format] docker start ' + name + ' failed:', e.message);
          return false;
        }
      }
      if (!(await _ollamaReady(3000))) {
        console.log('[format] Ollama unreachable, attempting auto-start...');
        if (await _dockerStart('ollama')) {
          // Wait for /api/tags to respond — cold-start ~5-10s
          const _deadline = Date.now() + 30000;
          while (Date.now() < _deadline) {
            if (await _ollamaReady(2000)) { console.log('[format] Ollama is up'); break; }
            await new Promise(r => setTimeout(r, 1500));
          }
        }
      }

      // ── Step 1: enhance caption ──────────────────────────────────────────
      let _cap = _origCaption;
      try {
        const _capPrompt = `Rewrite this music description as an ACE-Step generation prompt.\n\nFORMAT (mandatory):\n- Comma-separated tags only. NO sentences, no prose, no explanations.\n- 5-8 tags total (more is worse).\n- Tag order: [genre+era], [specific instruments], [mood/texture], [tempo BPM].\n- Use lowercase except proper nouns and BPM values.\n- ASCII characters only. No Chinese, Japanese, Korean, Cyrillic, or accented characters.\n\nINSTRUMENT NAMING (high impact):\n- For cinematic/orchestral: name SPECIFIC instruments (tremolo strings, low brass, timpani, sub bass, french horn, cello) — not generic "orchestral".\n- For electronic: name synth types (analog pad, FM bass, granular texture).\n- For rock/pop: name the kit (gritty guitar, driving bass, snare-heavy drums).\n- Generic "cinematic" alone makes the model fall back to alt-rock — instruments push it back into the right space.\n\nVOCAL HANDLING:\n- If the description mentions vocal qualities (baritone, raspy, breathy, falsetto, male, female, husky, powerful, smoky, nasal, gritty, whispered, mumble, rapper), include those EXACT words verbatim. Do NOT substitute synonyms.\n- If no vocals are described OR the description says "instrumental"/"no vocals"/"no lyrics", include the literal tag "instrumental" in the output.\n\nAVOID:\n- Contradictory tags ("dark" + "uplifting", "slow" + "energetic").\n- Vague mood adjectives without an instrument anchor.\n- The phrases "this song", "the music", or any meta-commentary.\n\nOutput ONLY the comma-separated tag string. No preamble. No quotes. No thinking aloud.\n\nDescription: ${_origCaption}`;
        const _capRes = await fetch(_ollamaUrl, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: _model, prompt: _capPrompt, stream: false, keep_alive: 0, options: { num_predict: 1024 } }),
          signal: AbortSignal.timeout(90_000),
        });
        if (_capRes.ok) {
          const _capData = await _capRes.json() as any;
          let _capGen = (_capData.response || '').trim();
          _capGen = _capGen.replace(/<think[^>]*>[\s\S]*?<\/think>/gi, '').trim();
          _capGen = _capGen.replace(/[^\x00-\x7F\n\r\t]/g, '').trim();
          if (_capGen.length > 20) { _cap = _capGen; console.log('[format] Caption enhanced:', _cap.slice(0, 80)); }
        }
      } catch (e: any) {
        console.warn('[format] Ollama caption failed, keeping original:', e.message);
      }

      // ── Step 2: detect intent — UNION of original + enhanced caption ─────
      // Why both: Gemma sometimes drops vocal-quality words ("baritone") during
      // rewrite. The original caption is the user's true intent signal; the
      // enhanced caption catches cases where the user typed something vague
      // and Gemma added "instrumental" tag.
      const _hasVocal = _vKW.test(_origCaption) || _vKW.test(_cap);
      const _hasInst = _iKW.test(_origCaption) || _iKW.test(_cap);
      const _isInstrumental = _hasInst && !_hasVocal;
      const _wantsVocal = _hasVocal && !_isInstrumental;

      // ── Step 3: lyrics ───────────────────────────────────────────────────
      // Decision tree:
      //   user provided lyrics?
      //     yes + has 3+ section markers (final draft) → preserve
      //     yes + short / no sections (seed)            → expand into full song
      //     no  + instrumental intent                   → '[instrumental]'
      //     no  + vocal intent                          → Gemma writes from scratch
      //     no  + ambiguous                             → '[instrumental]' default
      const _sectionRe = /\[(Intro|Verse|Pre-Chorus|Chorus|Bridge|Outro|Hook|Refrain)\b/gi;
      const _userSectionCount = (_userLyrics.match(_sectionRe) || []).length;
      const _userLyricsIsSeed = _userLyrics.length > 0 && _userSectionCount < 3;
      let _finalLyrics = '';
      if (_userLyrics && !_userLyricsIsSeed) {
        // User provided full lyrics — preserve verbatim
        _finalLyrics = _userLyrics;
        console.log('[format] Lyrics: user-provided full draft preserved (' + _userSectionCount + ' sections)');
      } else if (_userLyricsIsSeed) {
        // User provided a seed — expand into full song via Gemma
        try {
          const _expandPrompt = `Expand this seed into a ${_durSec}-second (~${_durMinStr}-minute) song lyrics paced for natural singing.\n\nThe seed below is the user's idea, hook, theme, or partial verse. Build a full song around it.\n\nCRITICAL TIMING RULE:\n- A ${_durSec}-second song fits roughly ${_minLines}-${_maxLines} sung lines TOTAL. Stay UNDER ${_hardCap} lines or vocals rush out of sync.\n- Repeat [Chorus] sections share the SAME lyrics — write chorus content ONCE in its first [Chorus] block, then on repeats just include the [Chorus] marker alone.\n\nFORMAT:\n- ASCII English only. No Chinese, Japanese, Korean, Cyrillic, or accented characters.\n- Section structure (line counts are MAX):\n   [Intro] — 2-4 lines\n   [Verse 1] — 6-8 lines\n   [Pre-Chorus] — 2-4 lines\n   [Chorus] — 4-6 lines (chorus content here)\n   [Verse 2] — 6-8 lines\n   [Pre-Chorus] — (marker only)\n   [Chorus] — (marker only)\n   [Bridge] — 4-6 lines\n   [Chorus] — (marker only)\n   [Outro] — 2-4 lines\n- KEEP the seed's theme/mood/specific lines verbatim where natural.\n- Each line: 6-10 syllables, sung-able.\n- Match the music style — if a vocal quality is in the style, write lines that suit that voice.\n- Real sung content. No metadata. No stage directions. No preamble or thinking aloud.\n- Output ONLY the full lyrics with section markers.\n\nMusic style: ${_cap}\n\nSeed (preserve where possible):\n${_userLyrics}`;
          const _expRes = await fetch(_ollamaUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: _model, prompt: _expandPrompt, stream: false, keep_alive: 0, options: { num_predict: 3072 } }),
            signal: AbortSignal.timeout(180_000),
          });
          if (_expRes.ok) {
            const _expData = await _expRes.json() as any;
            let _expanded = (_expData.response || '').trim();
            _expanded = _expanded.replace(/<think[^>]*>[\s\S]*?<\/think>/gi, '').trim();
            _expanded = _expanded.replace(/[^\x00-\x7F\n\r\t]/g, '').trim();
            const _hasSec = /\[(Intro|Verse|Chorus|Bridge|Outro)\b/i.test(_expanded);
            const _noTagText = _expanded.replace(/\[[^\]]+\]/g, '').trim();
            if (_hasSec && _noTagText.length > 200) {
              _finalLyrics = _expanded;
              console.log('[format] Lyrics: seed expanded to full song (' + _expanded.split('\n').length + ' lines)');
            }
          }
        } catch (e: any) {
          console.warn('[format] Ollama lyrics-expand failed:', e.message);
        }
        // Expansion failed → preserve seed verbatim (safer than empty)
        if (!_finalLyrics) {
          _finalLyrics = _userLyrics;
          console.log('[format] Lyrics: seed-expand fallback to verbatim user input');
        }
      } else if (_isInstrumental) {
        _finalLyrics = '[instrumental]';
        console.log('[format] Lyrics: [instrumental] (instrumental track)');
      } else if (_wantsVocal) {
          try {
            const _lyricsPrompt = `Write ${_durSec}-second (~${_durMinStr}-minute) song lyrics paced for natural singing — NOT padded.\n\nCRITICAL TIMING RULE:\n- A ${_durSec}-second song fits roughly ${_minLines}-${_maxLines} sung lines TOTAL. Going over makes vocals rush out of sync with the music. Stay UNDER ${_hardCap} lines.\n- Repeat [Chorus] sections share the SAME lyrics — write the chorus ONCE inside its first [Chorus] block, then on repeats just include the [Chorus] marker on its own line (no new lyrics under it). This is how producers indicate "play the chorus again."\n\nFORMAT:\n- ASCII English only. No Chinese, Japanese, Korean, Cyrillic, or accented characters.\n- Section structure (line counts are MAX, not min):\n   [Intro] — 2-4 lines\n   [Verse 1] — 6-8 lines\n   [Pre-Chorus] — 2-4 lines\n   [Chorus] — 4-6 lines (write the chorus content here)\n   [Verse 2] — 6-8 lines\n   [Pre-Chorus] — (marker only — repeats above)\n   [Chorus] — (marker only — repeats)\n   [Bridge] — 4-6 lines\n   [Chorus] — (marker only — repeats)\n   [Outro] — 2-4 lines\n- Each line: 6-10 syllables, naturally sung-able. Long words count more.\n- Match the genre and vocal style. If a vocal quality (baritone, raspy, breathy, falsetto, husky, smoky, gritty, soulful, passionate, melodic, whispered, etc.) is in the style, write lines that suit that voice and breath capacity.\n- Real sung content — vivid, on-theme, internally rhymed where natural. NOT placeholder stage directions.\n- DO NOT echo the music description text. DO NOT include genre tags or instrument names in the lyrics.\n- Output ONLY the lyrics with section markers. No preamble, no explanation, no thinking aloud, no quotes around the result.\n\nUser's original description: ${_origCaption}\nEnhanced style tags: ${_cap}`;
            const _lyrRes = await fetch(_ollamaUrl, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ model: _model, prompt: _lyricsPrompt, stream: false, keep_alive: 0, options: { num_predict: 3072 } }),
              signal: AbortSignal.timeout(240_000),
            });
            if (_lyrRes.ok) {
              const _lyrData = await _lyrRes.json() as any;
              let _generated = (_lyrData.response || '').trim();
              _generated = _generated.replace(/<think[^>]*>[\s\S]*?<\/think>/gi, '').trim();
              _generated = _generated.replace(/[^\x00-\x7F\n\r\t]/g, '').trim();
              const _noTags = _generated.replace(/\[[^\]]+\]/g, '').trim();
              // Validate: must have section markers, real content, AND not echo the caption
              const _hasSections = /\[(Intro|Verse|Chorus|Bridge|Outro)\b/i.test(_generated);
              const _echoesCaption = _cap.length > 30 && _generated.includes(_cap.slice(0, 30));
              if (_hasSections && _noTags.length > 100 && !_echoesCaption) {
                _finalLyrics = _generated;
                console.log('[format] Lyrics generated (', _generated.split('\n').length, 'lines):', _finalLyrics.slice(0, 60) + '...');
              } else {
                console.warn('[format] Lyrics validation failed (sections=' + _hasSections + ', textLen=' + _noTags.length + ', echo=' + _echoesCaption + ')');
              }
            }
          } catch (e: any) {
            console.warn('[format] Ollama lyrics failed:', e.message);
          }
          // Vocal but Gemma failed → structured scaffold (still better than empty for ACE-Step vocal mode)
          if (!_finalLyrics) {
            _finalLyrics = '[Intro]\n\n[Verse 1]\n\n[Pre-Chorus]\n\n[Chorus]\n\n[Verse 2]\n\n[Pre-Chorus]\n\n[Chorus]\n\n[Bridge]\n\n[Chorus]\n\n[Outro]';
            console.log('[format] Lyrics: scaffold fallback (vocal track, Gemma unavailable)');
          }
      } else {
        // Ambiguous intent (no clear vocal/instrumental keyword) — default to instrumental
        _finalLyrics = '[instrumental]';
        console.log('[format] Lyrics: [instrumental] (ambiguous intent, defaulting)');
      }

      res.json({
        caption: _cap,
        lyrics: _finalLyrics,
        bpm: d.bpm,
        duration: Number(d.duration) || _durSec,
        key_scale: d.key_scale,
        time_signature: d.time_signature,
        vocal_language: _isInstrumental ? 'instrumental' : 'en',
        vocalLanguage: _isInstrumental ? 'instrumental' : 'en',
      });
      return;
    } catch (fetchErr: any) {
      // Only fall back to Python spawn on network errors (service not yet reachable)
      if (fetchErr?.name !== 'AbortError' && (fetchErr?.code === 'ECONNREFUSED' || fetchErr?.cause?.code === 'ECONNREFUSED')) {
        console.warn('[Format] REST API unreachable, falling back to Python spawn');
      } else {
        console.error('[Format] REST API request failed:', fetchErr?.message);
        res.status(500).json({ success: false, error: fetchErr?.message || 'Format request failed' });
        return;
      }
    }

    // Fallback: Python spawn (only reached when REST API is unreachable)
    const { spawn } = await import('child_process');
    const ACESTEP_DIR = process.env.ACESTEP_PATH || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../ACE-Step-1.5');
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const SCRIPTS_DIR = path.join(__dirname, '../../scripts');
    const FORMAT_SCRIPT = path.join(SCRIPTS_DIR, 'format_sample.py');
    const pythonPath = resolvePythonPath(ACESTEP_DIR);

    const args = [FORMAT_SCRIPT, '--caption', caption, '--json'];
    if (lyrics) args.push('--lyrics', lyrics);
    if (bpm && bpm > 0) args.push('--bpm', String(bpm));
    if (duration && duration > 0) args.push('--duration', String(duration));
    if (keyScale) args.push('--key-scale', keyScale);
    if (timeSignature) args.push('--time-signature', timeSignature);
    if (temperature !== undefined) args.push('--temperature', String(temperature));
    if (topK && topK > 0) args.push('--top-k', String(topK));
    if (topP !== undefined) args.push('--top-p', String(topP));
    if (lmModel) args.push('--lm-model', lmModel);
    if (lmBackend) args.push('--lm-backend', lmBackend);

    console.log(`[Format] Fallback spawn: ${pythonPath} ${args.join(' ')}`);
    const result = await new Promise<{ success: boolean; data?: any; error?: string }>((resolve) => {
      const proc = spawn(pythonPath, args, {
        cwd: ACESTEP_DIR,
        env: { ...process.env, ACESTEP_PATH: ACESTEP_DIR },
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => { stdout += data.toString(); });
      proc.stderr.on('data', (data) => { stderr += data.toString(); });

      proc.on('close', (code) => {
        if (code === 0 && stdout) {
          const lines = stdout.trim().split('\n');
          let jsonStr = '';
          for (let i = lines.length - 1; i >= 0; i--) {
            if (lines[i].startsWith('{')) { jsonStr = lines[i]; break; }
          }
          try {
            const parsed = JSON.parse(jsonStr || stdout);
            resolve({ success: true, data: parsed });
          } catch {
            console.error('[Format] Failed to parse stdout:', stdout.slice(0, 500));
            resolve({ success: false, error: 'Failed to parse format result' });
          }
        } else {
          console.error(`[Format] Process exited with code ${code}`);
          if (stdout) console.error('[Format] stdout:', stdout.slice(0, 1000));
          if (stderr) console.error('[Format] stderr:', stderr.slice(0, 1000));
          resolve({ success: false, error: stderr || stdout || `Format process exited with code ${code}` });
        }
      });

      proc.on('error', (err) => {
        console.error('[Format] Spawn error:', err.message);
        resolve({ success: false, error: err.message });
      });
    });

    if (result.success && result.data) {
      res.json(result.data);
    } else {
      console.error('[Format] Python error:', result.error);
      res.status(500).json({ success: false, error: result.error });
    }
  } catch (error) {
    console.error('[Format] Route error:', error);
    res.status(500).json({ error: (error as Error).message });
  }
});

export default router;
