# Cubane Studio Fork of ace-step-ui

This is Owen's (`owolfson`) fork of [`fspecii/ace-step-ui`](https://github.com/fspecii/ace-step-ui), being expanded into **Cubane Studio** — an integrated music + music-video generation app.

## Why a fork?

Upstream `fspecii/ace-step-ui` ships a Gradio-backed UI. We swapped the backend to a custom **`acestep-server`** running ACE-Step v1.5's native FastAPI (`api_server.py`), which gives:

- Multi-LoRA registry with hot model swap (~10s vs 3-min restart)
- Full `/v1/*` API surface (training, dataset management, etc.)
- Proper `/docs` Swagger UI
- Live model inventory (vs the broken Gradio `/v1/models` that returns `{name:"unknown"}`)

These changes were originally maintained as a 1269-line `patch-source.mjs` regex-rewrite script applied at Docker build time. This fork **eliminates that patchwork** by bringing the changes in as real source commits we own and can review with `git diff`.

## What changed vs upstream

| File | Change |
|---|---|
| `server/src/services/gradio-client.ts` | Replaced with `NativeAceStepClient` shim — translates Gradio `predict('/endpoint', args)` calls into native `/v1/*` HTTP requests. Includes auto-swap to LoRA's training base model on `/load_lora`. |
| `server/src/services/acestep.ts` | `handle_file()` stubbed (returns plain path strings — native API doesn't use Gradio file-blob refs) |
| `server/src/routes/generate.ts` | `/format` endpoint enriched with Ollama-driven Gemma lyric generation + dynamic line-count budget computed from requested duration (replaces hardcoded 3-minute target). `/models` endpoint queries acestep-server's `/v1/model_inventory` for live load state. |
| `server/src/routes/lora.ts` | New `GET /list` endpoint walks `/app/checkpoints/loras/` to populate the LoRA picker UI. |
| `server/src/routes/songs.ts` | MP3 tagging hooks on song INSERT/PATCH + `/:id/cover` upload endpoint. |
| `server/src/services/mp3tagging.ts` | New file — wraps `node-id3` for setting cover art + ID3 tags on generated MP3s. |
| `server/src/index.ts` | Strip CSP middleware (avoids breaking inline scripts injected for Cubane pickers), SPA static serving, `/api/voices/list` endpoint + `/voices` static mount for chatterbox-tts voice library. |
| `index.html` | Dark-mode init script + LoRA picker / Cover picker / Voice picker overlays (Cubane Studio extensions). |
| `branding/` | Cubane Studio favicon + assets. |

## Backend

Cubane fork talks to **`acestep-server`** at `http://acestep-server:7860` (or `host.docker.internal:7861` from the UI container's perspective). See `/mnt/user/appdata/acestep-server/` on the Unraid host.

## Upstream tracking

```bash
git remote add upstream https://github.com/fspecii/ace-step-ui.git
git fetch upstream main
git diff upstream/main..HEAD -- server/src/services/gradio-client.ts   # see what's different
```

When upstream pushes new fixes:
```bash
git fetch upstream && git rebase upstream/main
# resolve conflicts in patched files; commit
```

## What's NOT in this commit (next steps)

- One missing patch: `services/acestep.ts` XL-tuned default values (model-aware `inferenceSteps`/`guidanceScale`/`shift`/`cfgIntervalEnd`/`useAdg` based on `ACESTEP_ACTIVE_MODEL`). Upstream code moved; needs re-application against current `acestep.ts`.
- ComfyUI music video pipeline integration (planned next phase).
- Multi-LoRA stacking UI (currently single-slot).
