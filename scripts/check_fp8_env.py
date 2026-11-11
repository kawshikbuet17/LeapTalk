"""Check that this machine/env can run the FP8 DiT path (run on the GPU box before benchmarking).

Usage: python scripts/check_fp8_env.py            # exit 0 = ready, 1 = not ready
Checks: torch + CUDA, GPU compute capability >= 8.9, torchao importable with the FP8 config classes,
and a tiny FP8 Linear forward (eager + torch.compile) matching bf16 within tolerance.
"""
from __future__ import annotations

import sys


def main() -> int:
    ok = True
    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: no CUDA GPU")
        return 1
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU 0: {torch.cuda.get_device_name(0)}, compute capability {cc[0]}.{cc[1]}")
    if cc < (8, 9):
        print("FAIL: FP8 tensor cores need compute capability >= 8.9 (RTX 40xx, L4/L40, H100)")
        ok = False

    try:
        import torchao
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_
        print(f"torchao {torchao.__version__}: Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_ OK")
    except Exception as e:  # ImportError or a binary/ABI mismatch with this torch
        print(f"FAIL: torchao import: {type(e).__name__}: {e}")
        print('      try: pip install "torchao==0.12.0"  (or 0.11.0 if 0.12 does not match torch 2.7.1)')
        return 1
    if not ok:
        return 1

    torch.manual_seed(0)
    lin = torch.nn.Sequential(torch.nn.Linear(1536, 8960)).to("cuda", torch.bfloat16).eval()
    x = torch.randn(4, 1024, 1536, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        ref = lin(x)
        quantize_(lin, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
        out = lin(x)
        out_c = torch.compile(lin)(x)
    for name, y in (("eager", out), ("compiled", out_c)):
        rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"FP8 Linear {name}: relative error vs bf16 = {rel:.4f}")
        if not rel < 0.1:
            print(f"FAIL: FP8 {name} output too far from bf16")
            ok = False
    print("READY" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
