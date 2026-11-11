# Running the FP8 DiT configs (RTX 4090)

**Goal:** lower the 512² stream RTF from ~0.45 (Lite + compile, bf16) toward 0.1–0.2. Most of the per-chunk time is the DiT (~300 ms of ~470 ms at 512, see BENCHMARK.md §3), and the DiT is mostly `nn.Linear` matmuls. The 4090 (cc 8.9) has FP8 tensor cores with ~2× the bf16 throughput. **Expected:** ~1.3–1.8× on the DiT step. This is an estimate, not a measurement.

## What FP8 mode does

- `fp8_utils.apply_fp8_to_dit` quantizes the `blocks.N.*` Linears with torchao `Float8DynamicActivationFloat8WeightConfig(PerRow())`:
  - weights: FP8 E4M3 with one scale per row;
  - activations: quantized per row on every call;
  - matmul: runs on FP8 tensor cores and returns bf16.
- **When it's applied:** after the LoRA `merge_and_unload()`, so the merged weights are quantized, and before `torch.compile`, so compile fuses the quantize kernels. **Always run FP8 with compile on**, because eager FP8 gains little.
- **Stays in bf16:** patch, time and text embeddings, the head, `audio_proj`, wav2vec2, and the TAE/VAE.
- **Scopes** (`--fp8_scope`):
  - `default`: all block Linears except `cross_attn.k/v`. Those see only ~32 audio tokens and carry the lip-sync signal.
  - `all`: every block Linear.
  - `ffn`: feed-forward Linears only. This is the most conservative.

## Where the switches are

| Entry point | Switch |
|---|---|
| `inference.py` / `inf.sh` | `--fp8 --fp8_scope default` / `FP8=1 FP8_SCOPE=default bash inf.sh` |
| `web_server.py` | `LEAPTALK_FP8=1`, `LEAPTALK_FP8_SCOPE=default` in `.env` |
| `bench/bench.py` | `--fp8 1 --fp8_scope default` (recorded in the result JSON) |
| `bench/ui_server.py` | `--precisions bf16,fp8` (both loaded, switch per request) or `--precisions fp8`; `--fp8_scope`; env `LEAPTALK_UI_PRECISIONS` |
| `bench/run_all.sh` | config `tag:lite:compile:size:repeats:fp8[:scope]` |

## Setup

```bash
conda activate leaptalk                       # /root/miniconda3/envs/leaptalk, torch 2.7.1
pip install "torchao==0.12.0"
python scripts/check_fp8_env.py              # must print READY (cc 8.9, torchao imports, FP8 Linear works)
# if torchao 0.12.0 fails to import against torch 2.7.1: pip install "torchao==0.11.0" and rerun
```

## Run

```bash
# FP8 vs the existing compiled bf16 baselines (lite_c1, lite_c1_256)
CONFIGS="lite_c1_fp8:1:on:512:2:1 lite_c1_fp8_256:1:on:256:2:1" bash bench/run_all.sh
python bench/make_report.py

# quality gate: same clip, bf16 vs FP8 (bench.py --save_videos 1 is the default)
python scripts/compare_videos.py bench/outputs/lite_c1 bench/outputs/lite_c1_fp8
python scripts/syncnet_score.py bench/outputs/lite_c1/stream_00.mp4 bench/outputs/lite_c1_fp8/stream_00.mp4
#   SyncNet needs eval/eval-lipsync/weights/{sfd_face.pth,syncnet_v2.model} (standard joonson/syncnet_python
#   weights) and pip install -r eval/eval-lipsync/requirements.txt. For 256² videos, lower the face
#   thresholds if no track is found, e.g. --min_face_size 50 --facedet_scale 0.5

# if quality fails, try the conservative scope
CONFIGS="lite_c1_fp8ffn:1:on:512:2:1:ffn" bash bench/run_all.sh

# look at it live: bf16 and FP8 in one server, Precision switch + "A/B compare" in the Generate tab
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python bench/ui_server.py --lite 1 --compile on --precisions bf16,fp8
```

Compare FP8 only against the **compiled** baselines (`lite_c1`, `lite_c1_256`). The first FP8 compile is cold and takes several minutes.

## Live testing in the web UI

With `--precisions bf16,fp8` the server keeps a bf16 DiT and an FP8 copy. Everything else (wav2vec, TAE, audio_proj weights) is the same, and a request switches between them in microseconds.
- **Precision** (bf16 | FP8) in the Generate tab applies to Stream, Non-stream and Echo. The live metrics show which one ran.
- **A/B compare** runs the same clip non-stream in bf16, then in FP8. It shows the two videos side by side with their RTF and the FP8 speedup. **Play both** plays them in sync, with audio from the bf16 one.
- **Cost:** about 1.4 GB more VRAM (Lite 512 peaks around 10 GB instead of 8.6 GB). Warmup covers both precisions, so expect a cold start of about 15–20 minutes with compile on. If the GPU is short of memory next to the Triton containers, use `--precisions fp8` alone.
- The server sets `torch._dynamo.config.cache_size_limit` to at least 32 in dual mode, so both compiled graphs stay compiled. If the FP8 speed looks like eager speed, check the log for dynamo recompile-limit warnings.

## Keep FP8 only if all of these hold

- [ ] Stream FPS is clearly higher than the matching baseline (512: > 59.5, 256: > 212.5), and the DiT stage time in the stage breakdown drops.
- [ ] Mean frame PSNR against bf16 is ≥ ~33 dB (`compare_videos.py` exits 0).
- [ ] Sync-C and Sync-D each change by ≤ 0.2 (`syncnet_score.py` with two files prints PASS).
- [ ] A side-by-side look in the UI shows no change in identity or mouth shape.

If speed passes but quality fails, rerun with `fp8_scope=ffn`. If `ffn` doesn't help either, FP8 isn't worth it here. Look at TensorRT or attention-side optimisations instead.

## Troubleshooting

- `FP8 needs compute capability >= 8.9`: wrong GPU (`CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
- `FP8 needs torchao ...`: install `torchao==0.12.0` into the env that runs the script.
- No speedup: check that compile is `on` and that the log line `FP8 applied to DiT: {...quantized_linears: N...}` shows N > 0.
