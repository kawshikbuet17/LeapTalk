"""LeapTalk benchmark on a single GPU.

Modes measured per audio clip (after warmup):
  * nonstream : whole clip -> wav2vec once -> all chunks (inference.py --audio_encode_mode once)
  * stream    : official streaming path (8 s wav2vec ring buffer, 1 chunk = 28 frames = 1.12 s),
                run back-to-back as fast as possible -> compute RTF / FPS / per-chunk latency
  * realtime  : streaming path with audio arriving at 1x wall-clock speed (like a live TTS feed),
                each chunk also encoded to fMP4 like the web demo -> TTFA and playback stalls

Usage: python bench/bench.py --tag lite_c0 --lite 1 --compile off
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics as st
import threading
import time
from pathlib import Path

import lt_core
from lt_core import ROOT, LeapTalkRunner


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class GpuMonitor:
    """Samples whole-device utilisation/memory via NVML (device is shared with other processes)."""

    def __init__(self, index: int = 0, period: float = 0.1):
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.period = period
        self.samples: list[tuple[int, float]] = []
        self._stop = threading.Event()
        self._t = None

    def name(self):
        n = self.nv.nvmlDeviceGetName(self.h)
        return n.decode() if isinstance(n, bytes) else n

    def snapshot(self):
        u = self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu
        m = self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2**30
        return u, m

    def start(self):
        self.samples.clear()
        self._stop.clear()

        def run():
            while not self._stop.is_set():
                self.samples.append(self.snapshot())
                time.sleep(self.period)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join()
        us = [u for u, _ in self.samples] or [0]
        ms = [m for _, m in self.samples] or [0]
        return {"util_mean": st.mean(us), "util_max": max(us), "mem_used_max_gb": max(ms)}


def read_list(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def run_stream(r: LeapTalkRunner, image, audio, save=None):
    state, prep = r.prepare(image)
    slices = r.split_slices(audio)
    stages, frames = [], []
    for sl in slices:
        f, d = r.stream_chunk(state, sl)
        stages.append(d)
        if save is not None:
            frames.append(f)
    if save is not None:
        import numpy as np
        r.save_mp4(np.concatenate(frames), audio, save)
    return prep, stages, len(slices) * r.slice_frames


def run_realtime(r: LeapTalkRunner, image, audio, tmpdir: Path, pipelined: bool):
    """Audio slice k becomes available at T0 + (k+1)*slice_seconds. Player starts at first ready chunk.

    pipelined=False: web-demo path — GPU compute then fMP4 encode serially in one thread.
    pipelined=True : fMP4 encode (x264 ultrafast) runs on a CPU worker while the GPU does the next chunk.
    """
    import concurrent.futures as cf
    state, prep = r.prepare(image)
    slices = r.split_slices(audio)
    d = r.slice_seconds
    pool = cf.ThreadPoolExecutor(max_workers=1) if pipelined else None
    futs, compute, frames_ready = [], [], []

    def enc_job(f, sl, k):
        e = r.encode_fmp4_chunk(f, sl, tmpdir / f"c{k:04d}.mp4", fast=True)
        return time.perf_counter() - t0, e

    t0 = time.perf_counter()
    for k, sl in enumerate(slices):
        avail = t0 + (k + 1) * d
        now = time.perf_counter()
        if now < avail:
            time.sleep(avail - now)
        f, st_ = r.stream_chunk(state, sl)
        compute.append(st_["total"])
        frames_ready.append(time.perf_counter() - t0)
        if pipelined:
            futs.append(pool.submit(enc_job, f, sl, k))
        else:
            e = r.encode_fmp4_chunk(f, sl, tmpdir / f"c{k:04d}.mp4")
            futs.append((time.perf_counter() - t0, e))
    done = [x.result() for x in futs] if pipelined else futs
    if pool:
        pool.shutdown()
    ready = [x[0] for x in done]
    encode = [x[1] for x in done]
    # player: chunk k due at play_start + k*d + accumulated stall
    play_start = ready[0]
    stall_total, stalls = 0.0, 0
    for k in range(1, len(ready)):
        due = play_start + k * d + stall_total
        if ready[k] > due + 1e-3:
            stalls += 1
            stall_total += ready[k] - due
    lat = [ready[k] - (k + 1) * d for k in range(len(ready))]  # slice-available -> chunk ready
    return {
        "prepare_s": prep,
        "chunks": len(slices),
        "ttfa_s": ready[0],
        "ttfa_frames_s": frames_ready[0],  # first raw frames on host, before video encoding
        "first_chunk_latency_s": lat[0],  # first slice available -> first chunk ready
        "chunk_latency_p50_s": pct(lat, 0.5),
        "chunk_latency_p95_s": pct(lat, 0.95),
        "chunk_latency_max_s": max(lat),
        "compute_p50_s": pct(compute, 0.5),
        "encode_p50_s": pct(encode, 0.5),
        "stalls": stalls,
        "stall_total_s": stall_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lite", type=int, default=1)
    ap.add_argument("--compile", default="off", choices=["off", "on", "auto"])
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--size", type=int, default=512, help="square output resolution, e.g. 256 or 512")
    ap.add_argument("--image", default=str(lt_core.DEFAULT_IMAGE))
    ap.add_argument("--audios", default=str(ROOT / "bench/audios.txt"))
    ap.add_argument("--warmup", default=str(ROOT / "bench/warmup_audios.txt"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--save_videos", type=int, default=1)
    ap.add_argument("--skip_realtime", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "bench/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = ROOT / "bench/outputs" / args.tag
    tmpdir = ROOT / "bench/runtime/rt" / args.tag
    tmpdir.mkdir(parents=True, exist_ok=True)

    mon = GpuMonitor(0)
    idle = [mon.snapshot() for _ in range(20) if time.sleep(0.1) is None]
    idle_util = st.mean(u for u, _ in idle)

    r = LeapTalkRunner(lite=bool(args.lite), compile_mode=args.compile, steps=args.steps)
    torch = r.torch
    r.settings.height = r.settings.width = args.size  # used by every prepare() below
    print(f"[{args.tag}] loaded in {r.load_seconds:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)

    audios = read_list(args.audios)
    wavs = {p: r.load_audio(p) for p in audios}
    warm = [r.load_audio(p) for p in read_list(args.warmup)]

    # ---------------- warmup: both paths, until per-chunk time is stable
    t = time.perf_counter()
    warm_chunk_times = []
    for w in warm:
        _, stg, _ = run_stream(r, args.image, w)
        warm_chunk_times += [x["total"] for x in stg]
        r.generate_full(r.prepare(args.image)[0], w)
    tmp_rt = tmpdir / "warm"
    tmp_rt.mkdir(exist_ok=True)
    s0, _ = r.prepare(args.image)
    for sl in r.split_slices(warm[0])[:3]:
        f, _ = r.stream_chunk(s0, sl)
        r.encode_fmp4_chunk(f, sl, tmp_rt / "w.mp4")
        r.encode_fmp4_chunk(f, sl, tmp_rt / "w2.mp4", fast=True)
    warmup_s = time.perf_counter() - t
    print(f"[{args.tag}] warmup {warmup_s:.1f}s, first/last warm chunk "
          f"{warm_chunk_times[0]:.3f}/{warm_chunk_times[-1]:.3f}s", flush=True)
    torch.cuda.reset_peak_memory_stats()

    rows = []
    mon.start()
    for i, p in enumerate(audios):
        a = wavs[p]
        dur = len(a) / r.sr
        row = {"audio": Path(p).name, "path": p, "duration_s": dur}

        # non-stream
        ns = []
        for _ in range(args.repeats):
            st_, _ = r.prepare(args.image)
            frames, d = r.generate_full(st_, a)
            ns.append(d)
        tot = [d["total"] for d in ns]
        row["nonstream"] = {
            "total_s": st.median(tot),
            "rtf": st.median(tot) / dur,
            "fps": frames.shape[0] / st.median(tot),
            "frames": int(frames.shape[0]),
            "audio_enc_s": st.median(d["audio_enc"] for d in ns),
            "chunk_p50_s": pct([c for d in ns for c in d["chunks"]], 0.5),
        }
        if args.save_videos:
            r.save_mp4(frames, a, vid_dir / f"nonstream_{i:02d}.mp4")

        # stream (as fast as possible)
        per = []
        for rep in range(args.repeats):
            save = vid_dir / f"stream_{i:02d}.mp4" if (args.save_videos and rep == 0) else None
            prep, stg, nfr = run_stream(r, args.image, a, save)
            per.append((prep, stg, nfr))
        all_tot = [x["total"] for _, stg, _ in per for x in stg]
        gen = st.median(sum(x["total"] for x in stg) for _, stg, _ in per)
        nfr = per[0][2]
        row["stream"] = {
            "gen_s": gen,
            "rtf": gen / dur,
            "fps": nfr / gen,
            "frames": nfr,
            "chunks": len(per[0][1]),
            "prepare_s": st.median(p for p, _, _ in per),
            "first_chunk_s": st.median(stg[0]["total"] for _, stg, _ in per),
            "chunk_p50_s": pct(all_tot, 0.5),
            "chunk_p95_s": pct(all_tot, 0.95),
            "chunk_max_s": max(all_tot),
            "stages_p50_s": {k: pct([x[k] for _, stg, _ in per for x in stg], 0.5)
                             for k in ["audio_enc", "denoise", "decode", "color+hist"]},
        }

        # real-time paced
        if not args.skip_realtime:
            row["realtime"] = run_realtime(r, args.image, a, tmpdir, pipelined=True)
            row["realtime_serial"] = run_realtime(r, args.image, a, tmpdir, pipelined=False)
        rows.append(row)
        rt = row.get("realtime", {})
        print(f"[{args.tag}] {i+1}/{len(audios)} {row['audio']} {dur:.1f}s | "
              f"nonstream RTF {row['nonstream']['rtf']:.3f} | stream RTF {row['stream']['rtf']:.3f} "
              f"chunk p50 {row['stream']['chunk_p50_s']*1000:.0f}ms | "
              f"RT TTFA {rt.get('ttfa_s', float('nan')):.3f}s stalls {rt.get('stalls', '-')} | "
              f"serial TTFA {row.get('realtime_serial', {}).get('ttfa_s', float('nan')):.3f}s "
              f"stalls {row.get('realtime_serial', {}).get('stalls', '-')}", flush=True)
    gpu = mon.stop()

    total_dur = sum(x["duration_s"] for x in rows)

    def agg(mode, key):
        return sum(x[mode][key] for x in rows)

    all_chunks = []
    summary = {
        "audio_total_s": total_dur,
        "nonstream": {
            "rtf": agg("nonstream", "total_s") / total_dur,
            "fps": agg("nonstream", "frames") / agg("nonstream", "total_s"),
            "latency_mean_s": agg("nonstream", "total_s") / len(rows),
        },
        "stream": {
            "rtf": agg("stream", "gen_s") / total_dur,
            "fps": agg("stream", "frames") / agg("stream", "gen_s"),
            "first_chunk_p50_s": st.median(x["stream"]["first_chunk_s"] for x in rows),
            "chunk_p50_s": st.median(x["stream"]["chunk_p50_s"] for x in rows),
            "chunk_p95_s": max(x["stream"]["chunk_p95_s"] for x in rows),
            "prepare_p50_s": st.median(x["stream"]["prepare_s"] for x in rows),
            "stages_p50_s": {k: st.median(x["stream"]["stages_p50_s"][k] for x in rows)
                             for k in rows[0]["stream"]["stages_p50_s"]},
        },
    }
    for key in ([] if args.skip_realtime else ["realtime", "realtime_serial"]):
        R = [x[key] for x in rows]
        summary[key] = {
            "ttfa_p50_s": st.median(x["ttfa_s"] for x in R),
            "ttfa_max_s": max(x["ttfa_s"] for x in R),
            "ttfa_frames_p50_s": st.median(x["ttfa_frames_s"] for x in R),
            "first_chunk_latency_p50_s": st.median(x["first_chunk_latency_s"] for x in R),
            "chunk_latency_p50_s": st.median(x["chunk_latency_p50_s"] for x in R),
            "chunk_latency_p95_s": max(x["chunk_latency_p95_s"] for x in R),
            "encode_p50_s": st.median(x["encode_p50_s"] for x in R),
            "stalls": sum(x["stalls"] for x in R),
            "stall_total_s": sum(x["stall_total_s"] for x in R),
        }

    result = {
        "tag": args.tag,
        "config": {"lite": bool(args.lite), "vae": "TAE (Lite)" if args.lite else "Wan2.1 VAE",
                   "compile": args.compile, "steps": args.steps, "dtype": "bf16",
                   "resolution": f"{args.size}x{args.size}", "fps": r.fps, "chunk_frames": r.slice_frames,
                   "chunk_seconds": r.slice_seconds, "repeats": args.repeats},
        "env": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "cuda": torch.version.cuda, "python": platform.python_version(),
                "gpu_idle_util_before_pct": idle_util},
        "load_s": r.load_seconds,
        "warmup_s": warmup_s,
        "warmup_chunk_times_s": warm_chunk_times,
        "peak_mem_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "gpu_during_bench": gpu,
        "cpu_loadavg_1m_end": __import__("os").getloadavg()[0],
        "summary": summary,
        "rows": rows,
    }
    (out_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
