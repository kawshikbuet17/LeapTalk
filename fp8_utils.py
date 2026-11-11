"""FP8 (E4M3) quantization of the LeapTalk DiT block Linears via torchao.

Weights are stored as FP8 with a per-row scale. Activations are quantized per row on every call
(dynamic), and the matmul runs on FP8 tensor cores (compute capability >= 8.9, e.g. RTX 4090 / H100)
with a bf16 output.

Call `apply_fp8_to_dit` AFTER `merge_and_unload()` (so the merged LoRA weights are quantized) and
BEFORE `torch.compile` (so compile can fuse the activation-quantize kernels; eager gains are small).

Only `blocks.N.*` nn.Linear layers are touched. Patch/time/text embeddings, the head, audio_proj,
wav2vec2 and the TAE/VAE stay in bf16.

Scopes:
  default : every block Linear except cross_attn.k / cross_attn.v (and k_img / v_img). Those see only
            the ~32 audio context tokens (no speed gain) and carry the lip-sync signal.
  all     : every block Linear.
  ffn     : only the feed-forward Linears (blocks.N.ffn.*), the most conservative choice.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
from loguru import logger

FP8_SCOPES = ("default", "all", "ffn")
TORCHAO_HINT = 'pip install "torchao==0.12.0"  (matches torch 2.7.1)'

_BLOCK_RE = re.compile(r"^blocks\.\d+\.")
_CROSS_KV_RE = re.compile(r"\.cross_attn\.(k|v|k_img|v_img)$")
_FFN_RE = re.compile(r"\.ffn\.\d+$")


def _unwrap(model: nn.Module) -> nn.Module:
    """Strip torch.compile / PEFT wrappers down to the module that owns `.blocks`."""
    m = model
    for _ in range(4):
        if hasattr(m, "blocks"):
            return m
        if hasattr(m, "_orig_mod"):  # torch.compile OptimizedModule
            m = m._orig_mod
        elif hasattr(m, "get_base_model"):  # PEFT (should already be merged)
            m = m.get_base_model()
        elif hasattr(m, "model"):
            m = m.model
        else:
            break
    if not hasattr(m, "blocks"):
        raise RuntimeError(f"apply_fp8_to_dit: could not find `.blocks` on {type(model).__name__}")
    return m


def _selected(fqn: str, scope: str) -> bool:
    if not _BLOCK_RE.match(fqn):
        return False
    if scope == "all":
        return True
    if scope == "ffn":
        return bool(_FFN_RE.search(fqn))
    return not _CROSS_KV_RE.search(fqn)  # default


def check_fp8_support(device: str | torch.device = "cuda") -> None:
    """Raise RuntimeError with a clear reason if FP8 cannot run here."""
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 needs a CUDA GPU")
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(idx)
    if cc < (8, 9):
        name = torch.cuda.get_device_name(idx)
        raise RuntimeError(f"FP8 needs compute capability >= 8.9 (Ada/Hopper); {name} is {cc[0]}.{cc[1]}")
    try:
        import torchao  # noqa: F401
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"FP8 needs torchao with Float8DynamicActivationFloat8WeightConfig: {e}. {TORCHAO_HINT}") from e


def apply_fp8_to_dit(model: nn.Module, scope: str = "default", device: str | torch.device = "cuda") -> dict:
    """Quantize the DiT block Linears of `model` to FP8 in place. Returns a summary dict.

    `model` may be the merged DiT, or a torch.compile wrapper around it (compile-before-LoRA case;
    compilation is lazy, so the first forward traces the quantized modules).
    """
    if scope not in FP8_SCOPES:
        raise ValueError(f"fp8_scope must be one of {FP8_SCOPES}, got {scope!r}")
    check_fp8_support(device)
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

    inner = _unwrap(model)
    chosen: list[str] = []
    skipped: list[str] = []
    for fqn, mod in inner.named_modules():
        if not isinstance(mod, nn.Linear) or not _BLOCK_RE.match(fqn):
            continue
        # torch._scaled_mm needs both matmul dims divisible by 16
        if _selected(fqn, scope) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0:
            chosen.append(fqn)
        else:
            skipped.append(fqn)
    if not chosen:
        raise RuntimeError(f"apply_fp8_to_dit: no Linear layers selected for scope={scope!r}")

    chosen_set = set(chosen)
    params_before = sum(inner.get_submodule(n).weight.numel() for n in chosen)
    quantize_(
        inner,
        Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        filter_fn=lambda mod, fqn: fqn in chosen_set,
    )
    summary = {
        "scope": scope,
        "quantized_linears": len(chosen),
        "bf16_block_linears": len(skipped),
        "quantized_params_m": round(params_before / 1e6, 1),
    }
    logger.info(f"FP8 applied to DiT: {summary}")
    return summary
