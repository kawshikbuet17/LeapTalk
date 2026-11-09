"""Render markdown tables from bench/results/*.json (pasted into BENCHMARK.md)."""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
ORDER = ["lite_c0", "lite_c1", "pro_c0", "pro_c1", "lite_c0_256", "lite_c1_256", "pro_c0_256", "pro_c1_256"]


def label(r):
    c = r["config"]
    res = c.get("resolution", "512x512").split("x")[0]
    return f"{res}² {'Lite (TAE)' if c['lite'] else 'Pro (Wan VAE)'}{' + compile' if c['compile'] == 'on' else ''}"


def ms(x):
    return f"{x * 1000:.0f} ms"


def main():
    rs = {json.loads(f.read_text())["tag"]: json.loads(f.read_text()) for f in RES.glob("*.json")}
    rs = [rs[t] for t in ORDER if t in rs] + [v for k, v in rs.items() if k not in ORDER]
    out = []
    out.append("| Config | Stream RTF | Stream FPS | Non-stream RTF | Non-stream FPS | Peak VRAM | Load | Warmup |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rs:
        s = r["summary"]
        out.append(f"| {label(r)} | {s['stream']['rtf']:.3f} | {s['stream']['fps']:.1f} | {s['nonstream']['rtf']:.3f} | "
                   f"{s['nonstream']['fps']:.1f} | {r['peak_mem_reserved_gb']:.1f} GB | {r['load_s']:.0f} s | {r['warmup_s']:.0f} s |")
    out.append("")
    out.append("| Config | TTFA frames | TTFA A/V (pipelined) | Chunk latency p50 / p95 | Stalls | TTFA A/V (demo serial) | Stalls (serial) | fMP4 encode |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rs:
        rt, rs_ = r["summary"].get("realtime"), r["summary"].get("realtime_serial")
        if rt and rs_:
            out.append(f"| {label(r)} | {rt['ttfa_frames_p50_s']:.2f} s | {rt['ttfa_p50_s']:.2f} s | "
                       f"{ms(rt['chunk_latency_p50_s'])} / {ms(rt['chunk_latency_p95_s'])} | {rt['stalls']} ({rt['stall_total_s']:.2f} s) | "
                       f"{rs_['ttfa_p50_s']:.2f} s | {rs_['stalls']} ({rs_['stall_total_s']:.2f} s) | {ms(rt['encode_p50_s'])} |")
    out.append("")
    out.append("| Config | wav2vec (8 s window) | DiT denoise | VAE decode | colour + history | Total (incl. D2H) |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for r in rs:
        s = r["summary"]["stream"]
        st = s["stages_p50_s"]
        out.append(f"| {label(r)} | {ms(st['audio_enc'])} | {ms(st['denoise'])} | {ms(st['decode'])} | {ms(st['color+hist'])} | {ms(s['chunk_p50_s'])} |")
    out.append("")
    tag = sys.argv[1] if len(sys.argv) > 1 else None
    best = next((r for r in rs if r["tag"] == tag), None) or min(rs, key=lambda r: r["summary"]["stream"]["rtf"])
    out.append(f"Per clip — {label(best)}:\n")
    out.append("| # | Audio | Stream RTF | 1st chunk | Chunk p95 | TTFA frames (RT) | Non-stream total | Non-stream RTF |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, x in enumerate(best["rows"], 1):
        rt = x.get("realtime", {})
        out.append(f"| {i} | {x['duration_s']:.1f} s | {x['stream']['rtf']:.3f} | {ms(x['stream']['first_chunk_s'])} | "
                   f"{ms(x['stream']['chunk_p95_s'])} | {rt.get('ttfa_frames_s', float('nan')):.2f} s | "
                   f"{x['nonstream']['total_s']:.2f} s | {x['nonstream']['rtf']:.3f} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
