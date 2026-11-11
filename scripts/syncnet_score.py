#!/usr/bin/env python
"""SyncNet lip-sync scores (Sync-C / Sync-D) for one or more mp4 files with muxed audio.

Reuses the vendored pipeline in eval/eval-lipsync/script/ (S3FD face detection ->
face tracking -> 224x224 crops -> SyncNet). Sync-C is the confidence (higher is better),
Sync-D the minimum audio-visual feature distance (lower is better).

Usage:
    python scripts/syncnet_score.py a.mp4 [b.mp4 ...] [--weights_dir eval/eval-lipsync/weights] [--json out.json]

    # quality gate: bf16 vs FP8 (prints b - a deltas and PASS/FAIL)
    python scripts/syncnet_score.py bench/outputs/lite_c0/nonstream_00.mp4 \\
        bench/outputs/lite_c0_fp8/nonstream_00.mp4 --max_delta 0.2

Options:
    --weights_dir DIR   folder with sfd_face.pth and syncnet_v2.model
                        (default: eval/eval-lipsync/weights, which is gitignored)
    --device cuda|cpu   default: cuda if available
    --max_delta 0.2     with exactly 2 files: PASS iff |dSync-C| <= max_delta and |dSync-D| <= max_delta
    --min_track 50      minimum face-track length in frames (pipeline default; 2 s at 25 fps)
    --min_face_size 100 minimum mean face box size in px (lower it for 256x256 videos if no track is found)
    --facedet_scale 0.25  S3FD input scale (raise it for small videos)

Requirements: the deps of eval/eval-lipsync/requirements.txt (opencv, scipy, scenedetect==0.6.6,
python_speech_features, ffmpeg-python) and an ffmpeg binary. If `ffmpeg` is not on PATH, the
binary shipped with imageio-ffmpeg is exposed under that name for the run.

Exit code: 0 on success (and PASS when comparing two files), 1 on FAIL, 2 on setup errors
(missing weights/deps) or when a video yields no usable face track.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYNC_DIR = REPO / "eval" / "eval-lipsync" / "script"
DEFAULT_WEIGHTS = REPO / "eval" / "eval-lipsync" / "weights"
S3FD_NAME = "sfd_face.pth"
SYNCNET_NAME = "syncnet_v2.model"


def check_weights(wdir: Path):
    s3fd, sync = wdir / S3FD_NAME, wdir / SYNCNET_NAME
    missing = [p for p in (s3fd, sync) if not p.is_file()]
    if missing:
        msg = [
            "ERROR: SyncNet weights not found:",
            *[f"  missing: {p}" for p in missing],
            "",
            f"Place these two files in {wdir} (or pass --weights_dir):",
            f"  {S3FD_NAME}       S3FD face detector state_dict (detectors/s3fd/nets.py: S3FDNet)",
            f"  {SYNCNET_NAME}   SyncNet state_dict (SyncNetModel.py: S)",
            "This repo has no download link for them. They are the standard weights used by the",
            "original SyncNet code (joonson/syncnet_python, see its download_model.sh), which the",
            "vendored eval/eval-lipsync/script code expects at eval/eval-lipsync/weights/.",
        ]
        print("\n".join(msg), file=sys.stderr)
        sys.exit(2)
    return s3fd, sync


def ensure_ffmpeg(tmp_root: Path) -> None:
    """The vendored pipeline calls `ffmpeg` by name; fall back to imageio-ffmpeg's binary."""
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        print("ERROR: ffmpeg not on PATH and imageio-ffmpeg unavailable", file=sys.stderr)
        sys.exit(2)
    bin_dir = tmp_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / "ffmpeg"
    if not link.exists():
        os.symlink(exe, link)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def load_pipeline(s3fd: Path, sync: Path, device: str, overrides: dict):
    if str(SYNC_DIR) not in sys.path:
        sys.path.insert(0, str(SYNC_DIR))
    try:
        from syncnet_pipeline import SyncNetPipeline
    except ImportError as e:
        print(f"ERROR: could not import the vendored SyncNet pipeline from {SYNC_DIR}: {e}\n"
              f"Install its deps: pip install -r eval/eval-lipsync/requirements.txt", file=sys.stderr)
        sys.exit(2)
    cfg = {"s3fd_weights": str(s3fd), "syncnet_weights": str(sync), **overrides}
    return SyncNetPipeline(cfg, device=device)


def score_video(pipe, video: Path, tmp_root: Path) -> dict:
    work = Path(tempfile.mkdtemp(prefix=f"{video.stem}_", dir=tmp_root))
    try:
        # audio is read from the same mp4 (the pipeline takes it as a separate path)
        offsets, confs, dists, max_conf, min_dist, _s3fd_json, has_face = pipe.inference(
            video_path=str(video), audio_path=str(video), cache_dir=str(work)
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ok = bool(confs)
    best = confs.index(max(confs)) if ok else None
    return {
        "video": str(video),
        "ok": ok,
        "has_face": bool(has_face),
        "num_tracks": len(confs),
        "sync_c": float(max_conf) if ok else None,
        "sync_d": float(min_dist) if ok else None,
        "av_offset": int(offsets[best]) if ok else None,
        "per_track": [{"offset": int(o), "conf": float(c), "dist": float(d)}
                      for o, c, d in zip(offsets, confs, dists)],
    }


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="SyncNet Sync-C / Sync-D for mp4 files with muxed audio.")
    ap.add_argument("videos", nargs="+", type=Path, help="mp4 files (audio track required)")
    ap.add_argument("--weights_dir", type=Path, default=DEFAULT_WEIGHTS,
                    help=f"folder with {S3FD_NAME} and {SYNCNET_NAME} (default: {DEFAULT_WEIGHTS})")
    ap.add_argument("--json", type=Path, default=None, help="write results to this JSON file")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--max_delta", type=float, default=0.2,
                    help="with 2 files: max allowed |delta| for Sync-C and Sync-D (default 0.2)")
    ap.add_argument("--min_track", type=int, default=None, help="min face-track length in frames (default 50)")
    ap.add_argument("--min_face_size", type=int, default=None, help="min face size in px (default 100)")
    ap.add_argument("--facedet_scale", type=float, default=None, help="S3FD detection scale (default 0.25)")
    args = ap.parse_args()

    for v in args.videos:
        if not v.is_file():
            print(f"ERROR: not a file: {v}", file=sys.stderr)
            return 2
    s3fd, sync = check_weights(args.weights_dir.resolve())

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    overrides = {k: getattr(args, k) for k in ("min_track", "min_face_size", "facedet_scale")
                 if getattr(args, k) is not None}

    results = []
    with tempfile.TemporaryDirectory(prefix="syncnet_score_") as td:
        tmp_root = Path(td)
        ensure_ffmpeg(tmp_root)
        pipe = load_pipeline(s3fd, sync, device, overrides)
        for v in args.videos:
            print(f"[syncnet] scoring {v}", file=sys.stderr)
            r = score_video(pipe, v.resolve(), tmp_root)
            if not r["ok"]:
                print(f"WARNING: no usable face track in {v} (has_face={r['has_face']}); "
                      f"try --min_face_size / --facedet_scale / --min_track", file=sys.stderr)
            results.append(r)

    w = max(10, max(len(Path(r["video"]).name) for r in results))
    print()
    print(f"{'video':<{w}} {'Sync-C':>8} {'Sync-D':>8} {'offset':>7} {'tracks':>7}")
    print("-" * (w + 34))
    for r in results:
        print(f"{Path(r['video']).name:<{w}} {fmt(r['sync_c']):>8} {fmt(r['sync_d']):>8} "
              f"{r['av_offset'] if r['av_offset'] is not None else '-':>7} {r['num_tracks']:>7}")

    out = {"device": device, "weights_dir": str(args.weights_dir), "results": results}
    code = 0 if all(r["ok"] for r in results) else 2

    if len(results) == 2:
        a, b = results
        if a["ok"] and b["ok"]:
            dc, dd = b["sync_c"] - a["sync_c"], b["sync_d"] - a["sync_d"]
            passed = abs(dc) <= args.max_delta and abs(dd) <= args.max_delta
            print("-" * (w + 34))
            print(f"delta (b - a): Sync-C {dc:+.3f}  Sync-D {dd:+.3f}  (max |delta| {args.max_delta})")
            print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
            out["delta"] = {"sync_c": dc, "sync_d": dd, "max_delta": args.max_delta,
                            "verdict": "PASS" if passed else "FAIL"}
            code = 0 if passed else 1
        else:
            print("VERDICT: FAIL (could not score both videos)")
            out["delta"] = {"verdict": "FAIL", "reason": "missing score"}
            code = 1

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return code


if __name__ == "__main__":
    sys.exit(main())
