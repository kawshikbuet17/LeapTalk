#!/usr/bin/env python
"""Frame-by-frame quality comparison of two videos (e.g. bf16 vs FP8 DiT output).

Usage:
    # single pair
    python scripts/compare_videos.py A.mp4 B.mp4 [--json out.json]

    # directory mode: pairs *.mp4 files with the same basename
    python scripts/compare_videos.py bench/outputs/lite_c0 bench/outputs/lite_c0_fp8 [--json out.json]

Options:
    --min_psnr 33.0              PASS threshold on mean full-frame PSNR (dB)
    --mouth_box 0.3,0.55,0.7,0.85
                                 mouth crop as fractions x1,y1,x2,y2 of width/height
    --no_ssim                    skip SSIM even if scikit-image is installed

For each pair, up to min(len_A, len_B) frames are compared as uint8 RGB. Reported:
frame count, mean/min/p5 PSNR, mouth-region mean PSNR, and mean SSIM (only when
scikit-image is importable). A is treated as the reference. If resolutions differ,
B is resized to A's size (with a warning).

Exit code 0 if every pair has mean PSNR >= --min_psnr, 1 otherwise (also 1 on errors
or if no pairs were found).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # older imageio
    import imageio  # type: ignore

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # skimage is optional
    _ssim = None

PSNR_CAP = 100.0  # identical frames (mse == 0) are reported as this value
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return PSNR_CAP
    return min(PSNR_CAP, 10.0 * math.log10(255.0 ** 2 / mse))


def to_rgb(fr: np.ndarray) -> np.ndarray:
    fr = np.asarray(fr)
    if fr.ndim == 2:
        fr = np.stack([fr] * 3, axis=-1)
    if fr.shape[-1] == 4:
        fr = fr[..., :3]
    if fr.dtype != np.uint8:
        fr = np.clip(fr, 0, 255).astype(np.uint8)
    return fr


def resize_to(fr: np.ndarray, hw) -> np.ndarray:
    from PIL import Image  # pillow is an imageio dependency

    return np.asarray(Image.fromarray(fr).resize((hw[1], hw[0]), Image.BICUBIC))


def mouth_slice(h: int, w: int, box):
    x1, y1, x2, y2 = box
    xs, xe = int(round(x1 * w)), int(round(x2 * w))
    ys, ye = int(round(y1 * h)), int(round(y2 * h))
    xe, ye = max(xe, xs + 1), max(ye, ys + 1)
    return slice(ys, ye), slice(xs, xe)


def compare_pair(path_a: Path, path_b: Path, mouth_box, use_ssim: bool) -> dict:
    ra = imageio.get_reader(str(path_a))
    rb = imageio.get_reader(str(path_b))
    psnrs, mouth_psnrs, ssims = [], [], []
    warnings = []
    shape_a = shape_b = None
    n_a = n_b = 0
    try:
        it_a, it_b = iter(ra), iter(rb)
        while True:
            try:
                fa = next(it_a)
            except (StopIteration, IndexError):
                fa = None
            try:
                fb = next(it_b)
            except (StopIteration, IndexError):
                fb = None
            if fa is not None:
                n_a += 1
            if fb is not None:
                n_b += 1
            if fa is None and fb is None:
                break
            if fa is None or fb is None:
                continue  # keep draining the longer one to count its frames
            fa, fb = to_rgb(fa), to_rgb(fb)
            if shape_a is None:
                shape_a, shape_b = fa.shape, fb.shape
                if shape_a != shape_b:
                    msg = (f"resolution differs: {path_a.name} {shape_a[1]}x{shape_a[0]} vs "
                           f"{path_b.name} {shape_b[1]}x{shape_b[0]}; resizing B to A")
                    warn(msg)
                    warnings.append(msg)
            if fb.shape != fa.shape:
                fb = resize_to(fb, fa.shape[:2])
            psnrs.append(psnr(fa, fb))
            ys, xs = mouth_slice(fa.shape[0], fa.shape[1], mouth_box)
            mouth_psnrs.append(psnr(fa[ys, xs], fb[ys, xs]))
            if use_ssim:
                ssims.append(float(_ssim(fa, fb, channel_axis=-1, data_range=255)))
    finally:
        ra.close()
        rb.close()

    if n_a != n_b:
        msg = f"frame count differs: {path_a.name}={n_a} vs {path_b.name}={n_b}; compared {min(n_a, n_b)}"
        warn(msg)
        warnings.append(msg)

    res = {
        "a": str(path_a),
        "b": str(path_b),
        "frames_a": n_a,
        "frames_b": n_b,
        "frames_compared": len(psnrs),
        "resolution_a": list(shape_a[:2][::-1]) if shape_a else None,
        "resolution_b": list(shape_b[:2][::-1]) if shape_b else None,
        "psnr_mean": None,
        "psnr_min": None,
        "psnr_p5": None,
        "mouth_psnr_mean": None,
        "ssim_mean": None,
        "warnings": warnings,
    }
    if psnrs:
        arr = np.asarray(psnrs)
        res.update(
            psnr_mean=float(arr.mean()),
            psnr_min=float(arr.min()),
            psnr_p5=float(np.percentile(arr, 5)),
            mouth_psnr_mean=float(np.mean(mouth_psnrs)),
            ssim_mean=float(np.mean(ssims)) if ssims else None,
        )
    return res


def list_videos(d: Path) -> dict:
    return {p.name: p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS}


def build_pairs(a: Path, b: Path):
    if a.is_dir() and b.is_dir():
        va, vb = list_videos(a), list_videos(b)
        common = sorted(set(va) & set(vb))
        for name in sorted(set(va) - set(vb)):
            warn(f"only in {a}: {name}")
        for name in sorted(set(vb) - set(va)):
            warn(f"only in {b}: {name}")
        return [(va[n], vb[n]) for n in common]
    if a.is_file() and b.is_file():
        return [(a, b)]
    raise SystemExit(f"ERROR: expected two files or two directories, got {a} and {b}")


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def parse_box(s: str):
    try:
        box = [float(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("mouth_box must be 4 comma-separated floats")
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise argparse.ArgumentTypeError("mouth_box must be x1,y1,x2,y2 with 0<=x1<x2<=1, 0<=y1<y2<=1")
    return box


def main() -> int:
    ap = argparse.ArgumentParser(description="PSNR/SSIM comparison of two videos or two directories of videos.")
    ap.add_argument("a", type=Path, help="reference video (or directory), e.g. bf16 output")
    ap.add_argument("b", type=Path, help="test video (or directory), e.g. FP8 output")
    ap.add_argument("--json", type=Path, default=None, help="write results to this JSON file")
    ap.add_argument("--min_psnr", type=float, default=33.0, help="PASS threshold on mean PSNR in dB (default 33.0)")
    ap.add_argument("--mouth_box", type=parse_box, default=parse_box("0.3,0.55,0.7,0.85"),
                    help="mouth crop x1,y1,x2,y2 in fractions (default 0.3,0.55,0.7,0.85)")
    ap.add_argument("--no_ssim", action="store_true", help="skip SSIM even if scikit-image is available")
    args = ap.parse_args()

    use_ssim = _ssim is not None and not args.no_ssim
    if _ssim is None and not args.no_ssim:
        warn("scikit-image not importable; SSIM skipped")

    pairs = build_pairs(args.a, args.b)
    if not pairs:
        print("ERROR: no matching video pairs found", file=sys.stderr)
        return 1

    results = []
    for pa, pb in pairs:
        try:
            r = compare_pair(pa, pb, args.mouth_box, use_ssim)
        except Exception as e:  # unreadable file etc.
            warn(f"failed to compare {pa} vs {pb}: {e}")
            r = {"a": str(pa), "b": str(pb), "error": str(e), "psnr_mean": None}
        r["pass"] = r.get("psnr_mean") is not None and r["psnr_mean"] >= args.min_psnr
        results.append(r)

    hdr = f"{'video':<28} {'frames':>7} {'PSNR mean':>10} {'min':>7} {'p5':>7} {'mouth':>7} {'SSIM':>7}  verdict"
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        name = Path(r["a"]).name
        frames = r.get("frames_compared", 0)
        if "error" in r:
            print(f"{name:<28} {'-':>7} {'ERROR':>10}  {r['error']}")
            continue
        print(f"{name:<28} {frames:>7} {fmt(r['psnr_mean']):>10} {fmt(r['psnr_min']):>7} "
              f"{fmt(r['psnr_p5']):>7} {fmt(r['mouth_psnr_mean']):>7} {fmt(r['ssim_mean'], 4):>7}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")

    overall = all(r["pass"] for r in results)
    valid = [r for r in results if r.get("psnr_mean") is not None]
    summary = {
        "min_psnr": args.min_psnr,
        "mouth_box": args.mouth_box,
        "ssim_enabled": use_ssim,
        "pairs": len(results),
        "overall_psnr_mean": float(np.mean([r["psnr_mean"] for r in valid])) if valid else None,
        "overall_mouth_psnr_mean": float(np.mean([r["mouth_psnr_mean"] for r in valid])) if valid else None,
        "verdict": "PASS" if overall else "FAIL",
    }
    print("-" * len(hdr))
    print(f"overall mean PSNR {fmt(summary['overall_psnr_mean'])} dB, mouth {fmt(summary['overall_mouth_psnr_mean'])} dB "
          f"over {len(results)} pair(s); threshold {args.min_psnr:.2f} dB")
    print(f"VERDICT: {summary['verdict']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
        print(f"wrote {args.json}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
