"""LeapTalk benchmark viewer + live demo.

python bench/ui_server.py --lite 1 --compile on            # scheme/port/domain/cert from bench/.env
python bench/ui_server.py --scheme http --port 8075        # override the env

bench/.env (gitignored, see bench/.env.example):
  LEAPTALK_UI_SCHEME=https|http   LEAPTALK_UI_PORT=8075   LEAPTALK_UI_DOMAIN=<your domain>
  LEAPTALK_UI_SSL_CERT=<full-chain cert>   LEAPTALK_UI_SSL_KEY=<private key>
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soxr
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
ECHO_MAX_S = 600  # an echo session holds the GPU; cap it so a forgotten tab frees it


def load_env(path: Path) -> None:
    """Minimal KEY=VALUE loader; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


class ChunkSender:
    """GPU makes chunk k+1 while a CPU thread encodes chunk k; then sends JSON meta + fMP4 bytes."""

    def __init__(self, ws: WebSocket, r: LeapTalkRunner, t_req: float):
        self.ws, self.r, self.t_req = ws, r, t_req
        self.t0 = time.perf_counter()  # audio stream start (echo resets it at the first mic packet)
        self.out_dir = RUNTIME / f"st_{uuid.uuid4().hex[:10]}"
        self.out_dir.mkdir()
        self.compute_s, self.n = 0.0, 0
        self.q: asyncio.Queue = asyncio.Queue()
        self.task = asyncio.create_task(self._send())

    async def add(self, state, sl: np.ndarray, avail: float) -> None:
        r = self.r
        frames, stg = await on_gpu(r.stream_chunk, state, sl)
        self.compute_s += stg["total"]
        path = self.out_dir / f"c{self.n:04d}.mp4"
        fut = asyncio.get_running_loop().run_in_executor(ENC, r.encode_fmp4_chunk, frames, sl, path, True)
        await self.q.put(({"type": "chunk", "index": self.n, "frames": int(frames.shape[0]), "fps": r.fps,
                           "compute_s": stg["total"], "stages": stg, "available_at_s": avail,
                           "frames_ready_at_s": time.perf_counter() - self.t0}, fut, path))
        self.n += 1

    async def _send(self) -> None:
        while (item := await self.q.get()) is not None:
            meta, fut, path = item
            enc = await fut
            ready = time.perf_counter() - self.t0
            meta.update(encode_s=enc, ready_at_s=ready, latency_s=ready - meta["available_at_s"],
                        since_request_s=time.perf_counter() - self.t_req)
            await self.ws.send_text(json.dumps(meta))
            await self.ws.send_bytes(path.read_bytes())
            path.unlink(missing_ok=True)

    async def finish(self) -> None:
        await self.q.put(None)
        await self.task

    def cancel(self) -> None:
        if self.task.done():
            if not self.task.cancelled():
                self.task.exception()  # mark a send error (client gone) as handled
        else:
            self.task.cancel()
        try:  # best effort: chunks are sent already, or nobody is listening
            for f in self.out_dir.glob("*"):
                f.unlink(missing_ok=True)
            self.out_dir.rmdir()
        except OSError:  # an encode was still in flight after a disconnect
            pass


async def report_error(ws: WebSocket, e: Exception) -> None:
    try:
        await ws.send_text(json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"}))
    except Exception:
        pass


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    out = None
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
            out = ChunkSender(ws, r, t_req)
            for k, sl in enumerate(slices):
                if realtime:  # slice k is "available" once its audio has arrived at 1x speed
                    wait = out.t0 + (k + 1) * d - time.perf_counter()
                    if wait > 0:
                        await asyncio.sleep(wait)
                await out.add(state, sl, time.perf_counter() - out.t0)
            await out.finish()
            dur = len(audio) / r.sr
            await ws.send_text(json.dumps({"type": "end", "compute_total_s": out.compute_s,
                                           "rtf": out.compute_s / dur,
                                           "fps": len(slices) * r.slice_frames / out.compute_s,
                                           "wall_s": time.perf_counter() - out.t0}))
    except WebSocketDisconnect:
        return
    except Exception as e:  # surface errors to the page
        await report_error(ws, e)
    finally:
        if out:
            out.cancel()
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/ws/echo")
async def ws_echo(ws: WebSocket):
    """Live mic -> the avatar repeats it.

    Client sends {"image", "size", "sample_rate"}, then binary int16 mono PCM at sample_rate, then
    {"type": "stop"}. Audio is resampled to 16 kHz as it arrives; every 1.12 s becomes one chunk
    through the same path as /ws/stream.
    """
    await ws.accept()
    out = recv = None
    try:
        if gpu_lock.locked():
            raise RuntimeError("the GPU is busy with another session; try again in a moment")
        req = json.loads(await ws.receive_text())
        image_path, size = resolve_image(req.get("image")), resolve_size(req.get("size"))
        rate = int(req.get("sample_rate") or 0)
        if not 8000 <= rate <= 192000:
            raise ValueError(f"unsupported sample_rate {rate}")
        r = runner
        async with gpu_lock:
            t_req = time.perf_counter()
            state, prep = await on_gpu(r.prepare, image_path, size)
            await ws.send_text(json.dumps({"type": "start", "prepare_s": prep, "chunk_seconds": r.slice_seconds,
                                           "fps": r.fps, "size": size, "max_s": ECHO_MAX_S}))
            rs = soxr.ResampleStream(rate, r.sr, 1, dtype="float32")
            blocks: asyncio.Queue = asyncio.Queue()
            out = ChunkSender(ws, r, t_req)
            heard = 0.0
            S = r.slice_samples

            async def receive() -> str:
                nonlocal heard
                buf = np.zeros(0, np.float32)
                started = False

                def push(y: np.ndarray, last: bool = False) -> None:
                    nonlocal buf
                    buf = np.concatenate([buf, y])
                    while len(buf) >= S:
                        blocks.put_nowait((buf[:S].copy(), time.perf_counter() - out.t0))
                        buf = buf[S:]
                    if last and len(buf) > r.sr // 10:  # pad a tail longer than 0.1 s to a full chunk
                        blocks.put_nowait((np.pad(buf, (0, S - len(buf))), time.perf_counter() - out.t0))

                flush = lambda: push(rs.resample_chunk(np.zeros(0, np.float32), last=True), last=True)
                try:
                    while True:
                        msg = await ws.receive()
                        if msg["type"] == "websocket.disconnect":
                            return "disconnect"
                        if msg.get("bytes"):
                            if not started:
                                out.t0, started = time.perf_counter(), True
                            x = np.frombuffer(msg["bytes"], "<i2").astype(np.float32) / 32768.0
                            heard += len(x) / rate
                            push(rs.resample_chunk(x))
                            if heard >= ECHO_MAX_S:
                                flush()
                                return "time limit"
                        elif msg.get("text") and json.loads(msg["text"]).get("type") == "stop":
                            flush()
                            return "stopped"
                finally:
                    blocks.put_nowait(None)

            recv = asyncio.create_task(receive())
            while (item := await blocks.get()) is not None:
                if recv.done() and recv.result() == "disconnect":
                    break  # nobody is watching; do not burn GPU on the backlog
                await out.add(state, *item)
            reason = await recv
            if reason == "disconnect":
                return
            await out.finish()
            await ws.send_text(json.dumps({"type": "end", "reason": reason, "audio_s": heard, "chunks": out.n,
                                           "compute_total_s": out.compute_s,
                                           "rtf": out.compute_s / max(heard, 1e-6)}))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await report_error(ws, e)
    finally:
        for t in (recv, out):
            if t:
                t.cancel()
        try:
            await ws.close()
        except Exception:
            pass


app.mount("/outputs", StaticFiles(directory=BENCH / "outputs"), name="outputs")
app.mount("/runtime", StaticFiles(directory=RUNTIME), name="runtime")


def main():
    global runner, server_cfg
    load_env(BENCH / ".env")
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(env("LEAPTALK_UI_PORT", "8075")))
    ap.add_argument("--scheme", choices=["http", "https"], default=env("LEAPTALK_UI_SCHEME") or None,
                    help="default: https if LEAPTALK_UI_SSL_CERT/KEY are set, else http")
    ap.add_argument("--lite", type=int, default=1)
    ap.add_argument("--compile", default="off")
    args = ap.parse_args()
    cert, key = env("LEAPTALK_UI_SSL_CERT", ""), env("LEAPTALK_UI_SSL_KEY", "")
    scheme = args.scheme or ("https" if cert and key else "http")
    if scheme == "https" and not (cert and key and Path(cert).is_file() and Path(key).is_file()):
        sys.exit("https needs readable LEAPTALK_UI_SSL_CERT and LEAPTALK_UI_SSL_KEY (bench/.env); or use --scheme http")

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
    print(f"URL: {scheme}://{env('LEAPTALK_UI_DOMAIN') or '<host>'}:{args.port}/", flush=True)
    ssl = {"ssl_certfile": cert, "ssl_keyfile": key} if scheme == "https" else {}
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **ssl)


if __name__ == "__main__":
    main()
