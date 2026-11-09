"""LeapTalk benchmark viewer + live demo.

python bench/ui_server.py --port 8075 --lite 1 --compile on
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import time
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import lt_core
from lt_core import ROOT, LeapTalkRunner

BENCH = ROOT / "bench"
RUNTIME = BENCH / "runtime" / "ui"
UPLOADS = RUNTIME / "uploads"
for d in (RUNTIME, UPLOADS):
    d.mkdir(parents=True, exist_ok=True)
(BENCH / "outputs").mkdir(exist_ok=True)

app = FastAPI()

GPU = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
ENC = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="enc")
gpu_lock = asyncio.Lock()
runner: LeapTalkRunner | None = None
server_cfg: dict = {}
SIZES = (256, 512)  # square output resolutions offered in the UI


def resolve_size(size) -> int:
    size = int(size or 512)
    if size not in SIZES:
        raise HTTPException(400, f"size must be one of {SIZES}")
    return size


def audio_list():
    return [l.strip() for l in (BENCH / "audios.txt").read_text().splitlines() if l.strip()]


def resolve_audio(audio: str) -> Path:
    if audio.startswith("bench:"):
        return Path(audio_list()[int(audio.split(":", 1)[1])])
    p = UPLOADS / Path(audio).name
    if not p.exists():
        raise HTTPException(404, "audio not found")
    return p


def resolve_image(image: str | None) -> Path:
    if not image or image == "default":
        return lt_core.DEFAULT_IMAGE
    p = UPLOADS / Path(image).name
    if not p.exists():
        raise HTTPException(404, "image not found")
    return p


async def on_gpu(fn, *a):
    return await asyncio.get_running_loop().run_in_executor(GPU, fn, *a)


# ---------------------------------------------------------------- API
@app.get("/")
def index():
    return FileResponse(BENCH / "ui" / "index.html")


@app.get("/api/info")
def info():
    import soundfile as sf
    items = [{"id": f"bench:{i}", "name": Path(p).name, "duration_s": sf.info(p).duration}
             for i, p in enumerate(audio_list())]
    return {"audios": items, "server": server_cfg}


@app.get("/api/results")
def results():
    out = []
    for f in sorted((BENCH / "results").glob("*.json")):
        out.append(json.loads(f.read_text()))
    return JSONResponse(out)


@app.get("/media/bench_audio/{i}")
def bench_audio(i: int):
    return FileResponse(audio_list()[i])


@app.get("/media/default_image")
def default_image():
    return FileResponse(lt_core.DEFAULT_IMAGE)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "x").suffix.lower()[:6] or ".bin"
    name = f"{uuid.uuid4().hex}{ext}"
    (UPLOADS / name).write_bytes(await file.read())
    return {"id": name}


class NonStreamReq(BaseModel):
    audio: str
    image: str | None = None
    size: int = 512


@app.post("/api/nonstream")
async def nonstream(req: NonStreamReq):
    audio_path, image_path = resolve_audio(req.audio), resolve_image(req.image)
    size = resolve_size(req.size)

    def job():
        r = runner
        t0 = time.perf_counter()
        audio = r.load_audio(str(audio_path))
        state, prep = r.prepare(image_path, size)
        frames, d = r.generate_full(state, audio)
        t_enc = time.perf_counter()
        name = f"ns_{uuid.uuid4().hex[:10]}.mp4"
        r.save_mp4(frames, audio, RUNTIME / name)
        enc = time.perf_counter() - t_enc
        dur = len(audio) / r.sr
        return {
            "url": f"/runtime/{name}",
            "audio_s": dur,
            "frames": int(frames.shape[0]),
            "prepare_s": prep,
            "generate_s": d["total"],
            "encode_s": enc,
            "total_s": time.perf_counter() - t0,
            "rtf": d["total"] / dur,
            "fps": frames.shape[0] / d["total"],
            "chunks": len(d["chunks"]),
            "size": size,
        }

    async with gpu_lock:
        return await on_gpu(job)


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    try:
        req = json.loads(await ws.receive_text())
        audio_path, image_path = resolve_audio(req["audio"]), resolve_image(req.get("image"))
        realtime = bool(req.get("realtime", True))
        size = resolve_size(req.get("size"))
        r = runner
        async with gpu_lock:
            t_req = time.perf_counter()
            audio = r.load_audio(str(audio_path))
            state, prep = await on_gpu(r.prepare, image_path, size)
            slices = r.split_slices(audio)
            await ws.send_text(json.dumps({"type": "start", "chunks": len(slices), "prepare_s": prep,
                                           "chunk_seconds": r.slice_seconds, "fps": r.fps,
                                           "audio_s": len(audio) / r.sr, "realtime": realtime,
                                           "size": size}))
            d = r.slice_seconds
            t0 = time.perf_counter()  # audio stream start
            out_dir = RUNTIME / f"st_{uuid.uuid4().hex[:10]}"
            out_dir.mkdir()
            compute_sum = 0.0
            loop = asyncio.get_running_loop()
            send_q: asyncio.Queue = asyncio.Queue()

            async def sender():
                # chunks are encoded on a CPU worker while the GPU already works on the next one
                while (item := await send_q.get()) is not None:
                    meta, fut, path = item
                    enc = await fut
                    ready = time.perf_counter() - t0
                    meta.update(encode_s=enc, ready_at_s=ready, latency_s=ready - meta["available_at_s"],
                                since_request_s=time.perf_counter() - t_req)
                    await ws.send_text(json.dumps(meta))
                    await ws.send_bytes(path.read_bytes())

            send_task = asyncio.create_task(sender())
            for k, sl in enumerate(slices):
                if realtime:
                    wait = t0 + (k + 1) * d - time.perf_counter()
                    if wait > 0:
                        await asyncio.sleep(wait)
                avail = time.perf_counter() - t0
                frames, stg = await on_gpu(r.stream_chunk, state, sl)
                compute_sum += stg["total"]
                path = out_dir / f"c{k:04d}.mp4"
                fut = loop.run_in_executor(ENC, r.encode_fmp4_chunk, frames, sl, path, True)
                meta = {"type": "chunk", "index": k, "frames": int(frames.shape[0]), "fps": r.fps,
                        "compute_s": stg["total"], "stages": stg, "available_at_s": avail,
                        "frames_ready_at_s": time.perf_counter() - t0}
                await send_q.put((meta, fut, path))
            await send_q.put(None)
            await send_task
            dur = len(audio) / r.sr
            await ws.send_text(json.dumps({"type": "end", "compute_total_s": compute_sum,
                                           "rtf": compute_sum / dur,
                                           "fps": len(slices) * r.slice_frames / compute_sum,
                                           "wall_s": time.perf_counter() - t0}))
    except WebSocketDisconnect:
        return
    except Exception as e:  # surface errors to the page
        try:
            await ws.send_text(json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


app.mount("/outputs", StaticFiles(directory=BENCH / "outputs"), name="outputs")
app.mount("/runtime", StaticFiles(directory=RUNTIME), name="runtime")


def main():
    global runner, server_cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8075)
    ap.add_argument("--lite", type=int, default=1)
    ap.add_argument("--compile", default="off")
    args = ap.parse_args()

    runner = LeapTalkRunner(lite=bool(args.lite), compile_mode=args.compile)
    # warmup both paths so the first user request is not paying compile/cudnn autotune cost
    t = time.perf_counter()
    warm = runner.load_audio(str(ROOT / "assets/news.wav"))[: runner.slice_samples * 4]
    for size in SIZES:  # each resolution has its own compiled graph
        for _ in range(2):
            st, _ = runner.prepare(lt_core.DEFAULT_IMAGE, size)
            for sl in runner.split_slices(warm):
                f, _ = runner.stream_chunk(st, sl)
            runner.encode_fmp4_chunk(f, sl, RUNTIME / "warm.mp4", True)
            runner.generate_full(runner.prepare(lt_core.DEFAULT_IMAGE, size)[0], warm)
        print(f"warm {size}x{size} done at {time.perf_counter() - t:.0f}s", flush=True)
    server_cfg = {"lite": bool(args.lite), "compile": args.compile, "steps": 1, "sizes": list(SIZES),
                  "gpu": runner.torch.cuda.get_device_name(0),
                  "load_s": runner.load_seconds, "warmup_s": time.perf_counter() - t}
    print(f"ready: {server_cfg}", flush=True)
    print(f"URL: http://<host>:{args.port}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
