# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LeapTalk: real-time, audio-driven talking-head video generation (reference image + speech → video), 1 NFE per chunk. It is built on SoulX-FlashHead-1.3B (a Wan-style DiT) plus a LeapTalk LoRA and an `audio_proj` checkpoint. A Brownian-bridge ("Bridge Forcing") formulation turns the reference latent into the video latent. This repo is a fork: `bench/` and `BENCHMARK.md` add an RTX 4090 benchmark and a separate web UI on top of upstream.

There is no test suite, linter or build step. Everything runs on a CUDA GPU with weights under `./models/` (gitignored). Environment: conda `leaptalk`, Python 3.12, `torch==2.7.1`, then `pip install -r requirements.txt`. See README.md for the HuggingFace download commands.

## Commands

```bash
# Offline inference (paths come from env vars or edits in inf.sh; wraps torchrun inference.py)
CKPT_DIR=./models/SoulX-FlashHead-1_3B WAV2VEC_DIR=./models/wav2vec2-base-960h \
LORA_DIR=./models/leaptalk LITE=1 COMPILE=off bash inf.sh      # extra args are passed through to inference.py
# useful inference.py flags: --out, --height/--width, --audio_encode_mode stream|once, --max_chunks N, --compile off|on|auto

# Upstream web demo (Doubao realtime voice dialogue + LeapTalk); config in .env (copy .env.example)
python web_server.py                                            # http://localhost:7860

# Training (VividHead dataset under ./data/VividHead/{videos,audios}); all knobs are env vars in train.sh
bash train.sh

# Benchmark (fork addition)
bash bench/run_all.sh                       # CONFIGS="tag:lite:compile[:size[:repeats[:fp8[:scope]]]] ..." to pick configs
python bench/bench.py --tag lite_c0 --lite 1 --compile off --size 512 --repeats 1   # one config
python bench/make_report.py                 # tables from bench/results/*.json
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python bench/ui_server.py --lite 1 --compile on [--scheme http]
```

`bench/run_all.sh` is specific to the shared 4090 server: it hardcodes `/root/miniconda3/envs/leaptalk` and `docker pause`s two Triton containers during the run (unpausing on exit), and writes logs to `logs/`. Adjust before using it elsewhere. `bench/ui_server.py` reads `bench/.env` (copy `bench/.env.example`). It needs HTTPS for the mic features, and it warms up every resolution and fps at startup, which takes minutes when compile is on.

## Architecture

**Core inference lives in `inference.py`.** The other entry points import its private helpers instead of duplicating them: `_pick_latest_lora_dir`, `_pick_latest_audio_proj`, `_resolve_lite_tae_path`, `_build_infer_timesteps`, `_lora_checkpoint_needs_compiled_base`, `StreamParams`, and others. `leaptalk_stream.LeapTalkEngine._load_sync` imports them lazily. `bench/lt_core.LeapTalkRunner` wraps `LeapTalkEngine` (it calls `_load_sync()` and reads `_helpers`/`_pipeline`/`_sp` directly) and mirrors `_generate_chunk_sync` with CUDA-event timing added. If you change a helper's signature or the per-chunk logic in `inference.py`/`leaptalk_stream.py`, update `bench/lt_core.py` to match.

**Chunked autoregressive generation.** `frame_num=33`, `motion_frames_latent_num=2`, so each chunk produces 28 new frames (1.12 s at 25 fps). Per chunk:
1. wav2vec2 encodes a rolling 8 s audio window (`cached_audio_duration`) → `audio_proj` → audio context.
2. `_bridge_sample_one_chunk`: the source `x_t` is the history latent prefix (hard-clamped before and after every step) concatenated with a suffix taken from the fixed reference-image latent. The DiT predicts velocity, `vibt/scheduler.ViBTScheduler` (a UniPC subclass) steps it, and audio CFG runs only when `guidance_scale != 1`.
3. Decode, apply colour correction against the reference (`match_and_blend_colors_torch`), and update the motion history (`history_update_mode`, default `roundtrip` = re-encode the decoded tail).

Defaults live in `flash_head/configs/infer_params.yaml` and the `LeapTalkStreamSettings` dataclass.

**Lite vs Pro** is a choice of decoder. Lite (`--lite`/`LITE=1`) uses the TAE (`flash_head/wan/modules/tae.py`), whose checkpoint is auto-resolved from the LoRA dir. Pro uses the full Wan VAE. Only Lite is real-time at 512² on a 4090. At 256² both are real-time. See BENCHMARK.md for numbers.

**Compile handling is subtle.** `flash_head/src/pipeline/flash_head_pipeline.py` reads module globals `COMPILE_MODEL` / `COMPILE_VAE`. Callers set these before constructing `FlashHeadPipeline`. If the LoRA keys contain `._orig_mod.` (saved from a compiled base), the base must be compiled *before* the LoRA loads. Otherwise compile happens *after* `PeftModel.from_pretrained(...).merge_and_unload()`. The VAE is never compiled in Lite mode.

**FP8 (optional, `fp8_utils.py`).** `apply_fp8_to_dit` quantizes only the `blocks.N.*` Linears with torchao (per-row FP8 weights, dynamic FP8 activations). It needs cc ≥ 8.9 and `torchao==0.12.0`. It is hooked into both `LeapTalkEngine._load_sync` and `inference.py` at the same point: after `merge_and_unload()` and before the compile-after-LoRA step. Keep that ordering. The switches are `fp8`/`fp8_scope` on `LeapTalkStreamSettings`, `--fp8` on the CLIs, and `LEAPTALK_FP8` for `web_server.py`. With `fp8_dual=True` (ui_server `--precisions bf16,fp8`), the engine loads `audio_proj`, then makes a deep copy of the DiT and quantizes that copy, so `engine.models` holds both. `set_precision()` switches between them by setting `pipeline.model` (callers must hold the GPU lock). RUN_FP8.md has the benchmark and quality-gate procedure (`scripts/compare_videos.py`, `scripts/syncnet_score.py`).

**Import path quirk.** `inference.py` prepends its own dir (plus a parent and a sibling `SoulX-FlashHead` dir, if one exists) to `sys.path` so that `flash_head` resolves to this repo's copy. Run scripts from the repo root. `bench/*` scripts insert the repo root themselves.

**Serving layers.**
- `leaptalk_stream.py`: `LeapTalkEngine` (single-thread executor, asyncio locks; one GPU job at a time) and `LeapTalkTurnStreamer` (accepts PCM16 audio, emits fMP4 video chunks).
- `web_server.py`: the upstream FastAPI demo. It speaks the Doubao realtime dialogue binary WebSocket protocol (`encode_frame`/`decode_frame`) and serves `web/static/`.
- `bench/ui_server.py` + `bench/ui/index.html`: the fork's own UI with stream, non-stream and mic-echo modes, 15–30 fps, and a face crop. Streaming goes over one long `POST /api/stream` HTTP response, not a WebSocket, so it still works after the user accepts a TLS certificate warning. fMP4 encoding runs on a CPU thread so it overlaps the GPU's work.

**Training** (`train.py`, `FlashHeadARBridgeTrainer`): LoRA on the student (1 step) with a bridge loss, optional DMD against a teacher (4 steps) with a fake-score critic LoRA, LPIPS, and spatial face/lip loss weighting from `spatial_mask_utils.py` (mediapipe). `utils/bridge_utils.py` holds EMA, checkpoint and LR-schedule helpers.

**`eval/`** is vendored third-party evaluation code (FID/FVD/LPIPS/PSNR/SSIM, SyncNet lip sync), not part of the runtime.

## Gotchas

- Upstream `--use_face_crop` (`flash_head/utils/facecrop.py`) uses the mediapipe `solutions` API, which the pinned mediapipe 0.10.35 no longer has. It silently falls back to the full image. `bench/lt_core.crop_to_face` reimplements the same framing with OpenCV Haar.
- The model was trained at 25 fps. Other fps values run but their lip sync is unverified. A chunk is always 28 frames, so the audio per chunk changes with fps.
- `flash_head/src/modules/flash_head_model copy.py` is a stray duplicate. The real module is `flash_head_model.py`.
- Gitignored: `models/`, `data/`, `outputs/`, `bench/outputs/`, `bench/runtime/`, `web/runtime/`, `logs/`, `.env`, and all weight files (`*.pt`, `*.safetensors`, ...).
