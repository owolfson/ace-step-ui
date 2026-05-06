import { config } from '../config/index.js';

/**
 * NATIVE-API CLIENT for ACE-Step v1.5.
 *
 * Replaces the previous @gradio/client wrapper. Talks to the native FastAPI
 * server (`acestep-server` container running upstream ACE-Step's
 * api_server.py) via JSON REST instead of Gradio's positional-args predict()
 * + websocket events.
 *
 * Public interface mirrors the Gradio Client:
 *   client.predict(endpoint: string, args: any[]) -> { data: [...] }
 *
 * That way every call site (services/acestep.ts, routes/{lora,generate,training}.ts)
 * keeps working without modification — the call pattern is identical, only
 * the transport changed.
 */

interface PredictResult { data: any[]; }

class NativeAceStepClient {
  baseUrl: string;
  constructor(baseUrl: string) { this.baseUrl = baseUrl.replace(/\/+$/, ''); }

  private async _post(path: string, body: any, timeoutMs: number = 60_000): Promise<any> {
    const r = await fetch(this.baseUrl + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error('[native-client] POST ' + path + ' HTTP ' + r.status + ': ' + text.slice(0, 200));
    }
    return r.json();
  }

  private async _get(path: string, timeoutMs: number = 30_000): Promise<any> {
    const r = await fetch(this.baseUrl + path, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error('[native-client] GET ' + path + ' HTTP ' + r.status + ': ' + text.slice(0, 200));
    }
    return r.json();
  }

  private async _request(method: string, path: string, body: any, timeoutMs: number = 30_000): Promise<any> {
    const init: any = {
      method,
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(timeoutMs),
    };
    if (body !== undefined && body !== null) init.body = JSON.stringify(body);
    const r = await fetch(this.baseUrl + path, init);
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error('[native-client] ' + method + ' ' + path + ' HTTP ' + r.status + ': ' + text.slice(0, 200));
    }
    return r.json();
  }

  /**
   * Translate /generation_wrapper's 50 positional args into a GenerateMusicRequest body.
   * Field names verified against acestep/api/http/release_task_models.py.
   * Snake_case canonical with camelCase aliases supported via PARAM_ALIASES.
   */
  private _argsToGenerateBody(args: any[]): any {
    // Index map mirrors buildGradioArgs() positions in services/acestep.ts
    return {
      // Caption goes into 'prompt' (the GenerateMusicRequest field name)
      prompt: args[0] || '',
      lyrics: args[1] || '',
      bpm: args[2] && args[2] > 0 ? args[2] : null,
      key_scale: args[3] || '',
      time_signature: args[4] || '',
      vocal_language: args[5] || 'en',
      inference_steps: args[6] ?? 8,
      guidance_scale: args[7] ?? 7.0,
      use_random_seed: args[8] ?? true,
      seed: args[9] ? String(args[9]) : '-1',
      reference_audio_path: this._fileArgToPath(args[10]),
      audio_duration: args[11] && args[11] > 0 ? args[11] : null,
      batch_size: args[12] ?? 1,
      src_audio_path: this._fileArgToPath(args[13]),
      audio_code_string: args[14] || '',
      repainting_start: args[15] ?? 0.0,
      repainting_end: args[16] ?? null,
      // instruction: (omit to use server's DEFAULT_DIT_INSTRUCTION)
      audio_cover_strength: args[18] ?? 1.0,
      cover_noise_strength: args[19] ?? 0.0,
      task_type: args[20] || 'text2music',
      use_adg: args[21] ?? false,
      cfg_interval_start: args[22] ?? 0.0,
      cfg_interval_end: args[23] ?? 1.0,
      shift: args[24] ?? 3.0,
      infer_method: args[25] || 'ode',
      // (timesteps is custom_timesteps — keep null for default)
      timesteps: args[26] || null,
      audio_format: args[27] || 'mp3',
      lm_temperature: args[28] ?? 0.85,
      thinking: args[29] ?? false,
      lm_cfg_scale: args[30] ?? 2.5,
      lm_top_k: args[31] || null,
      lm_top_p: args[32] ?? 0.9,
      lm_negative_prompt: args[33] || 'NO USER INPUT',
      // CoT flags — args[34] use_cot_metas does NOT exist on the native API
      use_cot_caption: args[35] ?? true,
      use_cot_language: args[36] ?? true,
      // 37 skipped (Gradio State)
      constrained_decoding_debug: args[38] ?? false,
      allow_lm_batch: args[39] ?? true,
      // get_scores / get_lrc / score_scale — NOT on native API (Gradio-only)
      lm_batch_chunk_size: args[43] ?? 8,
      track_name: args[44] || null,
      track_classes: args[45] || null,
      // enable_normalization, normalization_db, latent_shift, latent_rescale —
      // NOT on the native API (Gradio-only). Defaults applied server-side.
      // autogen (args[50]) — not directly on API, may be set elsewhere.
    };
  }

  /**
   * Translate handle_file()-style upload references back to plain file paths.
   * The acestep-server container has /shared-audio mounted from ace-step-ui's
   * ui-audio, so any /audio/... URL becomes /shared-audio/... at the server.
   */
  private _fileArgToPath(arg: any): string | null {
    if (!arg) return null;
    // Already a string path
    if (typeof arg === 'string') {
      if (arg.startsWith('/audio/')) return '/shared-audio' + arg.slice('/audio'.length);
      return arg;
    }
    // Gradio FileData object {url, path, orig_name}
    if (typeof arg === 'object') {
      const u = arg.url || arg.path || '';
      if (u.startsWith('/audio/')) return '/shared-audio' + u.slice('/audio'.length);
      return u || null;
    }
    return null;
  }

  /**
   * Submit a generation job and poll for completion.
   * Native API is async-only:
   *   POST /release_task → {task_id, status:"queued"}
   *   POST /query_result {task_id_list:"[...]"} → status updates until terminal
   *   GET  /v1/audio?path=<path> to retrieve the audio bytes
   * Returns Gradio-style {data:[result]} where result mimics generation_wrapper output.
   */
  private async _generateAsync(body: any): Promise<any> {
    // Step 1: submit (responses wrapped in {data, code, error, ...})
    const submitRaw = await this._post('/release_task', body, 30_000);
    const submitData = submitRaw.data || submitRaw;
    const taskId = submitData.task_id || submitData.id;
    if (!taskId) throw new Error('[native-client] /release_task did not return a task_id: ' + JSON.stringify(submitRaw).slice(0, 200));

    // Step 2: poll /query_result (terminal status code: 0=queued, 1=running, 2=success, 3=failed-ish)
    const deadlineMs = Date.now() + 1200_000; // 20-min cap per generation
    const pollIntervalMs = 2000;
    while (Date.now() < deadlineMs) {
      await new Promise(r => setTimeout(r, pollIntervalMs));
      const pollRaw = await this._post('/query_result', { task_id_list: JSON.stringify([taskId]) }, 30_000);
      const pollData = (pollRaw.data || pollRaw)[0] || pollRaw.data || pollRaw;
      const status = pollData.status;
      // status is mapped to int by server: 2 = success, 3+ = error
      if (status === 2 || status === 'success' || status === 'completed') {
        return pollData;
      }
      if (status === 3 || status === 4 || status === 'failed' || status === 'error') {
        throw new Error('[native-client] generation failed: ' + (pollData.error || pollData.message || 'unknown'));
      }
      // else: queued / running — keep polling
    }
    throw new Error('[native-client] generation timed out after 20min');
  }

  /**
   * The Gradio-compatible predict interface. Translates known endpoints
   * to native /v1/* calls. Unknown endpoints throw — caller should be updated.
   */
  async predict(endpoint: string, args: any[]): Promise<PredictResult> {
    switch (endpoint) {
      case '/generation_wrapper': {
        const body = this._argsToGenerateBody(args);
        const result = await this._generateAsync(body);
        // Wrap response in Gradio-style {data: [...]} for caller compatibility.
        // The result object should have audio_path (or similar) the caller reads.
        return { data: [result] };
      }

      case '/load_lora': {
        // Optional 2nd arg: adapter_name (for multi-adapter registry)
        // Path normalization for native acestep-server:
        //  1. Legacy spaceinvaderone layout: /app/ACE-Step-1.5/checkpoints/...
        //     The native server uses /app/checkpoints/... (alias-mounted as
        //     belt-and-braces, but rewrite anyway).
        //  2. Native /v1/lora/load wants the LoRA *directory* (with
        //     adapter_config.json), not the .safetensors file. UI Browse
        //     pickers point at the file — strip to dirname.
        let _loraPath = String(args[0] || '');
        if (_loraPath.startsWith('/app/ACE-Step-1.5/checkpoints/')) {
          _loraPath = '/app/checkpoints/' + _loraPath.substring('/app/ACE-Step-1.5/checkpoints/'.length);
        }
        if (_loraPath.endsWith('.safetensors')) {
          const _lastSlash = _loraPath.lastIndexOf('/');
          if (_lastSlash >= 0) _loraPath = _loraPath.substring(0, _lastSlash);
        }

        // Auto-swap to the LoRA's training base model BEFORE loading.
        // Different ACE-Step variants have different hidden sizes — a LoRA
        // trained on v15-turbo will fail to load against xl-sft with shape
        // mismatches. Read adapter_config.json and swap if needed.
        try {
          const fs = await import('fs/promises');
          const path = await import('path');
          // The shim runs INSIDE ace-step-ui container, where checkpoints
          // are mounted at /app/ACE-Step-1.5/checkpoints/. Translate
          // /app/checkpoints/ -> /app/ACE-Step-1.5/checkpoints/ for local fs.
          let _localPath = _loraPath;
          if (_localPath.startsWith('/app/checkpoints/')) {
            _localPath = _localPath.replace('/app/checkpoints/', '/app/ACE-Step-1.5/checkpoints/');
          }
          const cfgRaw = await fs.readFile(path.join(_localPath, 'adapter_config.json'), 'utf8');
          const cfg = JSON.parse(cfgRaw);
          const baseRef = String(cfg.base_model_name_or_path || '').trim();
          // Strip "checkpoints/" prefix if present — server expects bare model name
          const baseName = baseRef.startsWith('checkpoints/') ? baseRef.substring('checkpoints/'.length) : baseRef;
          if (baseName) {
            // Init/swap if (a) nothing is loaded yet, OR (b) a different model is loaded
            try {
              const inv = await this._get('/v1/model_inventory', 5_000);
              const loaded = (inv.data?.models || []).find((m: any) => m.is_loaded);
              if (!loaded || loaded.name !== baseName) {
                const _from = loaded ? loaded.name : 'none (cold)';
                console.log(`[load_lora] init/swap: ${_from} -> ${baseName} (LoRA trained against ${baseName})`);
                await this._post('/v1/init', { model: baseName, init_llm: false }, 180_000);
              } else {
                console.log(`[load_lora] base model ${baseName} already loaded — no swap needed`);
              }
            } catch (e: any) {
              console.warn('[load_lora] auto-swap check failed:', e?.message);
            }
          }
        } catch (e: any) {
          // adapter_config.json missing or unreadable — skip auto-swap, let
          // the load attempt fail with the server's normal error if base
          // doesn't match.
          console.warn('[load_lora] could not read adapter_config.json:', e?.message);
        }

        const reqBody: any = { lora_path: _loraPath };
        if (args[1]) reqBody.adapter_name = args[1];
        try {
          const result = await this._post('/v1/lora/load', reqBody, 30_000);
          return { data: [result.data?.message || result.message || 'loaded'] };
        } catch (e: any) {
          // Idempotency: server returns 400 "Adapter name already in use" when
          // an adapter with the same name is already registered (e.g. previous
          // load attempt succeeded server-side, UI state is stale). Treat as
          // success — the adapter IS loaded, which is the user's intent.
          const msg = String(e?.message || '');
          if (msg.includes('already in use') || msg.includes('already loaded')) {
            console.log('[load_lora] adapter already loaded — treating as success');
            return { data: ['LoRA already loaded (idempotent)'] };
          }
          throw e;
        }
      }

      case '/unload_lora': {
        const result = await this._post('/v1/lora/unload', {}, 15_000);
        return { data: [result.data?.message || result.message || 'unloaded'] };
      }

      case '/set_lora_scale': {
        const reqBody: any = { scale: args[0] };
        if (args[1]) reqBody.adapter_name = args[1];
        const result = await this._post('/v1/lora/scale', reqBody, 10_000);
        return { data: [result.data?.message || 'ok'] };
      }

      case '/set_use_lora': {
        const result = await this._post('/v1/lora/toggle', { use_lora: args[0] }, 10_000);
        return { data: [result.data?.message || 'ok'] };
      }

      case '/load_random_simple_description': {
        // Native endpoint is /create_random_sample (no /v1/ prefix)
        try {
          const result = await this._post('/create_random_sample', { sample_type: 'simple_mode' }, 5_000);
          const sample = result.data || result;
          // The sample has fields like prompt/caption/lyrics — return a description string
          return { data: [sample.prompt || sample.caption || sample.global_caption || 'cinematic, dark, instrumental'] };
        } catch (e: any) {
          const fallbacks = [
            'cinematic orchestral, dark, brooding, tremolo strings, 70 bpm, instrumental',
            'edm, melodic, female vocals, soulful, 120 bpm',
            'dark trap, male baritone, gritty, 80 bpm',
            'reggae, dancehall, male vocal, 95 bpm, conscious',
            'soul ballad, female vocals, jazzy piano, 70 bpm',
          ];
          return { data: [fallbacks[Math.floor(Math.random() * fallbacks.length)]] };
        }
      }

      case '/load_existing_dataset_for_preprocess': {
        // Training-flow endpoint. Path on native API is unverified — best-guess.
        const result = await this._post('/v1/training/load-dataset', { dataset_path: args[0] }, 60_000);
        return { data: [result.data || result] };
      }

      // ── Training tab: load dataset manifest (.json) by path ────────────────
      case '/load_training_dataset': {
        // args[0] = dataset_path or tensor_dir
        const result = await this._post('/v1/dataset/load', { dataset_path: args[0] }, 60_000);
        const status = result.data?.message || result.message || 'loaded';
        return { data: [status] };
      }

      // ── Training tab: get one sample for preview/edit ──────────────────────
      case '/get_sample_preview': {
        // args[0] = sample idx
        const idx = Number(args[0]) || 0;
        const result = await this._get('/v1/dataset/sample/' + idx, 30_000);
        const s: any = result.data || result;
        // training.ts expects: [audio, filename, caption, genre, promptOverride,
        // lyrics, bpm, key, timesig, duration, language, instrumental, rawLyrics]
        return {
          data: [
            s.audio_path || s.audio || '',
            s.filename || '',
            s.caption || '',
            s.genre || '',
            s.prompt_override || 'Use Global Ratio',
            s.lyrics || '',
            s.bpm || 120,
            s.keyscale || s.key_scale || s.key || '',
            s.timesignature || s.time_signature || '',
            s.duration || 0,
            s.language || 'unknown',
            !!s.is_instrumental,
            s.raw_lyrics || s.lyrics || '',
          ],
        };
      }

      // ── Training tab: persist a sample edit ────────────────────────────────
      case '/save_sample_edit': {
        // args[0..9] = sampleIdx, caption, genre, promptOverride, lyrics, bpm,
        //              key, timeSignature, language, instrumental
        const idx = Number(args[0]) || 0;
        const body = {
          sample_idx: idx,
          caption: args[1] || '',
          genre: args[2] || '',
          prompt_override: args[3] || 'Use Global Ratio',
          lyrics: args[4] || '',
          bpm: Number(args[5]) || 120,
          keyscale: args[6] || '',
          timesignature: args[7] || '',
          language: args[8] || 'instrumental',
          is_instrumental: !!args[9],
        };
        // PUT /v1/dataset/sample/{idx}
        const result = await this._request('PUT', '/v1/dataset/sample/' + idx, body, 30_000);
        const status = result.data?.message || result.message || 'saved';
        // training.ts expects: [dataframe, editStatus]
        return { data: [null, status] };
      }

      // ── Training tab: kick off LoRA training ───────────────────────────────
      case '/training_wrapper': {
        // 13 positional args mapping to StartTrainingRequest:
        //   0  tensorDir
        //   1  rank        → lora_rank
        //   2  alpha       → lora_alpha
        //   3  dropout     → lora_dropout
        //   4  learningRate
        //   5  epochs      → train_epochs
        //   6  batchSize   → train_batch_size
        //   7  gradientAccumulation
        //   8  saveEvery   → save_every_n_epochs
        //   9  shift       → training_shift
        //   10 seed        → training_seed
        //   11 outputDir   → lora_output_dir
        //   12 resumeCheckpoint  (NOT in StartTrainingRequest schema; ignored)
        const body: any = {
          tensor_dir: args[0] || '',
          lora_rank: Number(args[1]) || 64,
          lora_alpha: Number(args[2]) || 128,
          lora_dropout: Number(args[3]) ?? 0.1,
          learning_rate: Number(args[4]) || 0.0001,
          train_epochs: Number(args[5]) || 10,
          train_batch_size: Number(args[6]) || 1,
          gradient_accumulation: Number(args[7]) || 4,
          save_every_n_epochs: Number(args[8]) || 5,
          training_shift: Number(args[9]) ?? 3.0,
          training_seed: Number(args[10]) ?? 42,
          lora_output_dir: args[11] || './lora_output',
        };
        const result = await this._post('/v1/training/start', body, 60_000);
        const d: any = result.data || result;
        // training.ts expects: [trainingProgress, trainingLog, lineplotData]
        return {
          data: [
            d.message || d.task_id || 'training started',
            d.log || '',
            d.metrics || null,
          ],
        };
      }

      // ── Training tab: stop in-flight training ──────────────────────────────
      case '/stop_training': {
        const result = await this._post('/v1/training/stop', {}, 15_000);
        const status = result.data?.message || result.message || 'stopped';
        return { data: [status] };
      }

      // ── Training tab: auto-label entire dataset ────────────────────────────
      case '/auto_label_all': {
        // args: [skipMetas, formatLyrics, transcribeLyrics, onlyUnlabeled]
        const body: any = {
          skip_metas: !!args[0],
          format_lyrics: !!args[1],
          transcribe_lyrics: !!args[2],
          only_unlabeled: !!args[3],
          batch_size: 1,
        };
        const result = await this._post('/v1/dataset/auto_label_async', body, 60_000);
        const d: any = result.data || result;
        // training.ts expects: [dataframe, status]
        return { data: [null, d.message || ('task_id: ' + (d.task_id || '?'))] };
      }

      // ── Training tab: initialize ACE-Step service for training session ─────
      case '/init_service_wrapper': {
        // args: [checkpoint, configPath, device, initLlm, lmModelPath, backend,
        //        useFlashAttention, offloadToCpu, ...]
        const body: any = {
          model: args[1] || args[0] || 'acestep-v15-xl-sft',
          init_llm: !!args[3],
        };
        if (args[4]) body.lm_model_path = args[4];
        const result = await this._post('/v1/init', body, 600_000);
        const status = result.data?.message || result.message || 'initialized';
        return { data: [status] };
      }

      // ── Training tab: export trained LoRA to safetensors ───────────────────
      case '/export_lora': {
        // args: [exportPath, loraOutputDir]
        const body: any = {
          export_path: args[0] || './lora_output/final_lora',
          lora_output_dir: args[1] || './lora_output',
        };
        const result = await this._post('/v1/training/export', body, 60_000);
        const status = result.data?.message || result.message || 'exported';
        return { data: [status] };
      }

      // ── Training tab: import a previously-saved dataset (JSON manifest) ────
      case '/import_dataset': {
        // args: [datasetType, datasetPath]
        // Native server uses /v1/dataset/load for JSON imports
        const result = await this._post('/v1/dataset/load', { dataset_path: args[1] || args[0] }, 60_000);
        const status = result.data?.message || result.message || 'imported';
        return { data: [status] };
      }

      default:
        throw new Error('[native-client] Unknown endpoint: ' + endpoint + '. Add a case in NativeAceStepClient.predict() or update the caller.');
    }
  }
}

let clientInstance: NativeAceStepClient | null = null;

/**
 * Get the lazy-initialized native-API client. Caches a single instance.
 */
export async function getGradioClient(): Promise<NativeAceStepClient> {
  if (!clientInstance) {
    clientInstance = new NativeAceStepClient(config.acestep.apiUrl);
    console.log('[native-client] connected to ' + config.acestep.apiUrl);
  }
  return clientInstance;
}

/**
 * Reset the cached client. Forces a new connection on next use.
 */
export function resetGradioClient(): void {
  clientInstance = null;
}

/**
 * Health-check the native API. Returns true if /health responds 200.
 * Replaces the previous Gradio-availability multi-endpoint probe.
 */
export async function isGradioAvailable(): Promise<boolean> {
  try {
    const r = await fetch(config.acestep.apiUrl.replace(/\/+$/, '') + '/health', {
      signal: AbortSignal.timeout(5_000),
    });
    return r.ok;
  } catch {
    return false;
  }
}
