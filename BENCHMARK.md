# LeapTalk on RTX 4090 — benchmark

**Bottom line**

- **512×512:** only **Lite** (TAE decoder) is real-time. The best result is **RTF 0.45, ~60 FPS**, with first frames **~1.6 s** after audio starts. **Pro** (full Wan VAE) is **not real-time** (RTF ~1.4), because its VAE decode alone takes ~1 s per 1.12 s chunk.
- **256×256:** everything is real-time. Lite + compile reaches **RTF 0.13, ~213 FPS** streaming (~262 FPS non-stream), with first frames after **~1.27 s**. This matches the README's "up to 200 FPS". Pro also becomes real-time (RTF ~0.35).

## Setup

| | |
|---|---|
| GPU | RTX 4090 24 GB, driver 560.35, CUDA 12.6, torch 2.7.1, Python 3.12 |
| Model | SoulX-FlashHead-1.3B (Pro DiT) + LeapTalk LoRA (merged), 1 NFE, bf16, 25 fps, square 512² or 256² output |
| Streaming | 28 new frames per chunk = **1.12 s of audio per chunk**; wav2vec re-encodes an 8 s audio window every chunk (official `leaptalk_stream.py` path) |
| Test set | 10 Bangla TTS clips (OmniVoice outputs), 2.8–20.8 s, 89.5 s in total |
| Warmup | 3 separate clips through every code path before timing; compile configs warmed until chunk time was stable |
| Repeats | Lite ×2, Pro ×1 (chunk times vary < 2 %); medians reported |
| GPU sharing | The two Triton containers on this GPU were **paused** during the runs. They still held ~13 GB of VRAM. |
| CPU load | Shared server. Load average was ~44 (on 40 cores) during the 512 runs and ~5–9 during the 256 runs. This mainly affects fMP4 encode times and TTFA A/V. GPU stage times compare fairly. |

**Terms used below**
- **RTF**: compute time ÷ audio duration. Below 1 means faster than real-time.
- **FPS**: video frames generated per second of compute.
- **TTFA**: time from when audio starts arriving at 1× speed until the first chunk is ready. It always includes the 1.12 s spent buffering the first chunk.
  - **frames**: raw frames are ready on the host.
  - **A/V**: an encoded, playable fMP4 chunk is ready.
- **Stall**: playback had to wait for a chunk.

## 1. Throughput

| Res | Config | Stream RTF | Stream FPS | Non-stream RTF | Non-stream FPS | Peak VRAM | Model load | Warmup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | Lite (TAE) | 0.500 | 53.8 | 0.489 | 55.5 | 8.6 GB | 26 s | 74 s |
| 512 | **Lite (TAE) + compile** | **0.452** | **59.5** | **0.463** | **58.6** | 8.7 GB | 28 s | 478 s |
| 512 | Pro (Wan VAE) | 1.455 | 18.5 | 1.464 | 18.5 | 9.9 GB | 27 s | 261 s |
| 512 | Pro (Wan VAE) + compile | 1.383 | 19.4 | 1.368 | 19.8 | 9.9 GB | 28 s | 324 s |
| 256 | Lite (TAE) | 0.171 | 157.6 | 0.150 | 181.1 | 5.2 GB | 15 s | 21 s |
| 256 | **Lite (TAE) + compile** | **0.127** | **212.5** | **0.104** | **261.8** | 5.2 GB | 15 s | 49 s |
| 256 | Pro (Wan VAE) | 0.403 | 66.7 | 0.377 | 72.0 | 5.4 GB | 15 s | 263 s |
| 256 | Pro (Wan VAE) + compile | 0.348 | 77.3 | 0.320 | 84.8 | 5.5 GB | 16 s | 118 s |

- **Stream**: the live path, chunk by chunk, processed as fast as possible.
- **Non-stream**: the whole clip at once, as in `inference.py --audio_encode_mode once`.
- The 256 compile warmups were short because the compile cache from earlier runs was reused. A cold start takes several minutes, as at 512.

## 2. Real-time streaming latency (audio fed at 1× speed)

| Res | Config | TTFA frames | TTFA A/V | Chunk latency p50 / p95 | Stall time (pipelined) | TTFA A/V (demo serial) | Stall time (demo serial) |
|---|---|---:|---:|---:|---:|---:|---:|
| 512 | Lite (TAE) | 1.63 s | 2.27 s | 1.16 / 1.20 s | 0.22 s | 2.42 s | 12.5 s |
| 512 | **Lite (TAE) + compile** | **1.58 s** | 2.29 s | 1.18 / 1.24 s | 0.52 s | 2.44 s | 14.2 s |
| 512 | Pro (Wan VAE) | 2.62 s | 3.29 s | 4.15 / 10.6 s | 38.3 s | 3.41 s | 87.6 s |
| 512 | Pro (Wan VAE) + compile | 2.55 s | 3.23 s | 3.80 / 9.87 s | 34.8 s | 3.36 s | 83.4 s |
| 256 | Lite (TAE) | 1.31 s | 1.84 s | 0.74 / 0.86 s | 0.90 s | 1.89 s | 0.95 s |
| 256 | **Lite (TAE) + compile** | **1.27 s** | **1.73 s** | 0.59 / 0.74 s | 0.46 s | 1.76 s | 0.87 s |
| 256 | Pro (Wan VAE) | 1.55 s | 2.17 s | 1.05 / 1.24 s | 0.64 s | 2.24 s | 0.92 s |
| 256 | Pro (Wan VAE) + compile | 1.50 s | 1.99 s | 0.85 / 1.01 s | 0.61 s | 2.01 s | 0.50 s |

Stall times are totals over all 89.5 s of audio.

- **Demo serial**: what the official web demo does. Each chunk is generated and then encoded to fMP4 in the same thread.
- **Pipelined**: chunk *k* is encoded on a CPU thread while the GPU generates chunk *k+1*. This takes the encode (0.4–0.7 s) off the critical path.
- Stall time under ~1 s in total is timing jitter of tens of milliseconds per chunk, which you won't notice. Only 512 Pro, and the serial path at 512, stall for real.
- At 256 the serial path stalls little too, because generation plus encoding fits inside the 1.12 s budget, helped by the quieter CPU during that run.

## 3. Where the time goes (per chunk, stream, median)

| Res | Config | wav2vec (8 s window) | DiT denoise | VAE decode | Colour + history | Total |
|---|---|---:|---:|---:|---:|---:|
| 512 | Lite (TAE) | 22 ms | 345 ms | 105 ms | 8 ms | 521 ms |
| 512 | Lite (TAE) + compile | 21 ms | 299 ms | 106 ms | 8 ms | 469 ms |
| 512 | Pro (Wan VAE) | 26 ms | 347 ms | 1017 ms | 9 ms | 1515 ms |
| 512 | Pro (Wan VAE) + compile | 24 ms | 301 ms | 987 ms | 9 ms | 1438 ms |
| 256 | Lite (TAE) | 17 ms | 119 ms | 15 ms | 4 ms | 176 ms |
| 256 | Lite (TAE) + compile | 18 ms | 74 ms | 16 ms | 4 ms | 133 ms |
| 256 | Pro (Wan VAE) | 17 ms | 122 ms | 229 ms | 4 ms | 409 ms |
| 256 | Pro (Wan VAE) + compile | 17 ms | 67 ms | 228 ms | 4 ms | 353 ms |

The real-time budget is 1120 ms per chunk. "Total" also includes CPU work and the copy to host (~20 ms at 256, ~40 ms at 512), so it is a bit more than the sum of the stages.
- 512 Lite uses about 42 % of the budget. 512 Pro exceeds it because of the Wan VAE decode alone.
- 256 Lite + compile uses about 12 %, and 256 Pro about 32–37 %.

## 4. Per clip (Lite + compile)

| # | Duration | Stream RTF 512 | Stream RTF 256 | TTFA frames 512 | TTFA frames 256 | Non-stream total 512 | Non-stream total 256 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.8 s | 0.498 | 0.146 | 1.59 s | 1.29 s | 1.38 s | 0.32 s |
| 2 | 3.7 s | 0.512 | 0.143 | 1.57 s | 1.26 s | 1.86 s | 0.40 s |
| 3 | 4.5 s | 0.417 | 0.129 | 1.59 s | 1.32 s | 1.85 s | 0.52 s |
| 4 | 5.6 s | 0.502 | 0.157 | 1.59 s | 1.28 s | 2.29 s | 0.63 s |
| 5 | 8.1 s | 0.464 | 0.130 | 1.59 s | 1.25 s | 3.61 s | 0.94 s |
| 6 | 9.4 s | 0.450 | 0.119 | 1.58 s | 1.25 s | 4.35 s | 0.89 s |
| 7 | 10.6 s | 0.446 | 0.127 | 1.58 s | 1.27 s | 4.69 s | 1.11 s |
| 8 | 11.5 s | 0.450 | 0.124 | 1.58 s | 1.26 s | 5.34 s | 1.39 s |
| 9 | 12.6 s | 0.445 | 0.138 | 1.58 s | 1.31 s | 5.98 s | 1.31 s |
| 10 | 20.8 s | 0.433 | 0.109 | 1.57 s | 1.25 s | 10.11 s | 1.77 s |

Short clips have slightly higher RTF because the last chunk is padded up to a full 1.12 s.

## Takeaways

- **For real-time at 512, use Lite.** It is 2× faster than real-time with steady ~0.5 s chunks. At 256, any config works.
- **TTFA has a floor of 1.12 s**: the first chunk of audio must arrive first. Compute adds ~0.47 s at 512 and ~0.15 s at 256 (Lite + compile).
- **256 is ~3–3.5× faster than 512** for Lite and ~3.6–4× for Pro (stream RTF). The DiT and the decoder both scale with pixel count. Visual quality was not measured; compare the videos in the UI.
- **`torch.compile` speeds up the DiT** by 13 % at 512 and ~40 % at 256. The cost is minutes of warmup on a cold start. It's worth it for a long-running server, not for one-off runs.
- **Encode off the GPU thread.** The demo's serial encode turns a real-time 512 model into a stuttering stream on a busy CPU.
- **The README's "up to 200 FPS" holds at 256×256** (213 FPS streaming, 262 FPS non-stream). At 512×512 it is ~60 FPS.

## Web UI

`bench/ui_server.py` serves one page with two tabs.

- **Benchmark**: the tables above, per-clip results and sample videos.
- **Generate**: pick 256² or 512², a frame rate, then a mode. The default reference image is `bench/avatar_512x512.png`, and you can upload your own. The benchmark itself used `assets/photo.jpg`.
  - **Stream**: chunk-by-chunk playback, with audio fed at 1× like live TTS.
  - **Non-stream**: the whole clip at once, then play.
  - **Echo (mic)**: say a sentence and stop talking. After a pause (default 1.5 s, adjustable from 0.3 to 5 s) the avatar repeats the whole sentence. The mic is ignored while the avatar talks, so its voice is not echoed back.
  - Audio for Stream and Non-stream can be a test clip, an uploaded file, or a mic recording.
  - **Real-time speed** (Stream and Echo, off by default): when on, the audio is fed at 1× like live TTS. When off, it is generated as fast as possible.
  - **Crop to face** (optional, off by default): crops a square around the face before generating. **Face ratio** (1.5 to 3.0, default 2.0) sets how much space is kept around the face, and a preview shows the crop. If no face is found, the full image is used.
    - Upstream's crop (`--use_face_crop`) uses an old mediapipe API that the pinned mediapipe 0.10.35 no longer has, so it silently falls back to the full image.
    - The UI uses OpenCV's built-in face detector instead, with the same framing as upstream.

```
 Echo:  mic ──> browser: detect speech, wait for the pause, resample to 16 kHz
                  │  WAV upload, then the normal stream path
                  v
        ui_server.py ──> 28-frame chunks -> DiT + VAE (GPU) -> h264/aac fMP4 (CPU thread)
                  │
                  v
        browser plays the chunks as they arrive (MSE)
```

Stream and Echo get their chunks over one long HTTPS response (`POST /api/stream`), not a WebSocket. This way they keep working when the browser has only accepted the certificate with a warning, which blocks WebSockets.

**Echo latency** (512², 25 fps, Lite + compile, 4.5 s sentence). The time is measured from the end of the pause to the first video chunk, and the upload takes ~70 ms.

| Real-time speed | Pause end → first chunk |
|---|---|
| off | ~1.2 s |
| on | ~2.3 s (waits for 1.12 s of "arriving" audio first) |

Add the pause itself to get the time from when you stop talking.

**Frame rate.** The model was trained at 25 fps. Other rates run, but their lip sync has not been checked.

A chunk is always 28 frames, so the compute per chunk stays the same. A lower fps puts more audio into each chunk and lowers the RTF. These numbers are stream mode in the UI server, median of 4 runs:

| fps | Audio per chunk | Stream RTF 512² | Stream RTF 256² |
|---:|---:|---:|---:|
| 15 | 1.87 s | 0.31 | 0.10 |
| 20 | 1.40 s | 0.37 | 0.13 |
| **25** | 1.12 s | 0.49 | 0.17 |
| 30 | 0.93 s | 0.57 | 0.19 |

Compute is ~520 ms per chunk at 512² and ~140 ms at 256². These RTFs are a little higher than in section 1, because the UI server and other jobs share the CPU.

**Settings** are in `bench/.env`. It is gitignored; copy `bench/.env.example` to create it.

| Variable | Meaning |
|---|---|
| `LEAPTALK_UI_SCHEME` | `https` or `http` |
| `LEAPTALK_UI_PORT` | port to serve on |
| `LEAPTALK_UI_DOMAIN` | domain shown in the startup URL |
| `LEAPTALK_UI_SSL_CERT`, `LEAPTALK_UI_SSL_KEY` | full-chain certificate and its key (https only) |

`--scheme` and `--port` override the file.

- **Mic needs HTTPS.** Browsers allow the mic only on https or localhost. Over plain http, Record and Echo are turned off.
- **One job at a time.** Requests share one GPU and wait for each other. Stopping a stream or echo reply frees the GPU in about 1 s.

## Reproduce

```bash
conda activate leaptalk            # torch 2.7.1 + requirements.txt
bash bench/run_all.sh              # 512: all 4 configs; pauses/unpauses the 4090 Triton containers
CONFIGS="lite_c0_256:1:off:256:2 lite_c1_256:1:on:256:2 pro_c0_256:0:off:256:1 pro_c1_256:0:on:256:1" \
  bash bench/run_all.sh            # 256: same matrix (format tag:lite:compile:size:repeats[:fp8[:scope]])
python bench/make_report.py        # tables from bench/results/*.json
cp bench/.env.example bench/.env  # web UI settings: scheme, port, domain, cert + key
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python bench/ui_server.py --lite 1 --compile on   # web UI; add --scheme http to skip TLS
```

Raw results are in `bench/results/*.json`, and the generated videos are in `bench/outputs/<config>/` (`*_256` for 256×256).
