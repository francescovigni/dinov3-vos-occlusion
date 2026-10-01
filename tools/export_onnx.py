"""Export the frozen DINOv3 backbone to ONNX, one static-shape file per input size.

Target is TensorRT 8.2 on a Jetson Nano (Maxwell, compute 5.3): FP16 at best, no INT8,
no on-device PyTorch (JetPack 4.6 tops out at Python 3.6). So the graph is built here and
only the engine is built there.

The exported graph is exactly what ``dvos.backbone.extract_features`` returns —
last-layer patch tokens, layer-normed, reshaped to BxCxH/16xW/16 — optionally with the
L2 normalisation that ``dvos.propagate`` applies anyway, folded in so the device side is
a plain matmul.

    python tools/export_onnx.py --out artifacts/onnx
    python tools/export_onnx.py --sizes 480x864 --no-l2 --opset 14

Self-check: ``python tools/export_onnx.py --check-only`` asserts the wrapper reproduces
extract_features on random input, which is the only thing that can silently go wrong here.
"""

from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from dvos.backbone import PATCH, extract_features, load_dinov3

# TensorRT 8.2 parses ONNX opset <= 13 and IR version <= 8. torch exports
# aten::scaled_dot_product_attention only from opset 14, and writes a newer IR version,
# so attention is replaced by its textbook form for the duration of the export and the
# IR version is pinned afterwards. Both are export-time only; the weights are untouched.
TRT82_IR_VERSION = 8

# 16:9-ish, all multiples of PATCH; 480x864 is the config's DAVIS size.
DEFAULT_SIZES = "480x864,384x672,320x576,256x448,192x336"


def _sdpa_explicit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """scaled_dot_product_attention in opset-13 ops. Only the unmasked eval path."""
    if attn_mask is not None or is_causal or enable_gqa:
        raise NotImplementedError("export path covers unmasked, non-causal attention only")
    s = (1.0 / math.sqrt(q.shape[-1])) if scale is None else scale
    return torch.softmax(q @ k.transpose(-2, -1) * s, dim=-1) @ v


@contextlib.contextmanager
def frozen_rope(backbone: torch.nn.Module, hp: int, wp: int):
    """Replace the RoPE table with constants for a fixed patch grid.

    Two reasons, both blocking. The grid size reaches ``rope_embed`` as a traced tensor, so
    its ``max(H, W)`` becomes a data-dependent ONNX ``If`` whose branches differ in shape and
    TensorRT 8.2 refuses to parse it. And the table is rebuilt once per block — twelve times
    per frame — for a value that never changes at a static input size. Constants kill both.
    The table is bf16 upstream, which TensorRT 8.2 cannot represent, so it is widened to fp32.
    """
    rope = getattr(backbone, "rope_embed", None)
    if rope is None:
        yield
        return
    with torch.no_grad():
        sin, cos = rope(H=hp, W=wp)
    sin, cos = sin.detach().float().clone(), cos.detach().float().clone()
    orig = rope.forward
    rope.forward = lambda *, H, W: (sin, cos)  # noqa: ARG005 - shape is fixed by construction
    try:
        yield
    finally:
        rope.forward = orig


@contextlib.contextmanager
def opset13_attention():
    orig = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa_explicit
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig


class PatchTokens(torch.nn.Module):
    """Frozen backbone -> BxCxhxw patch tokens, optionally L2-normalised over C."""

    def __init__(self, backbone: torch.nn.Module, l2: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.l2 = l2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
        return F.normalize(f, dim=1) if self.l2 else f


def parse_sizes(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        h, w = (int(v) for v in part.strip().lower().split("x"))
        if h % PATCH or w % PATCH:
            raise ValueError(f"{part}: both sides must be multiples of {PATCH}")
        out.append((h, w))
    return out


def check(model: PatchTokens, size: tuple[int, int]) -> None:
    """The wrapper must return what the pipeline's own extractor returns."""
    h, w = size
    x = torch.randn(1, 3, h, w)
    with torch.inference_mode():
        ref = extract_features(model.backbone, x)
        # Both export-time substitutions must leave the features alone.
        with opset13_attention(), frozen_rope(model.backbone, h // PATCH, w // PATCH):
            got = model(x)
    if model.l2:
        ref = F.normalize(ref, dim=1)
    assert got.shape == ref.shape == (1, 384, h // PATCH, w // PATCH), (got.shape, ref.shape)
    err = (got - ref).abs().max().item()
    assert err < 1e-4, f"wrapper drifted from extract_features: max abs err {err}"
    print(f"check {h}x{w}: ok (max abs err {err:.2e}, tokens {(h // PATCH) * (w // PATCH)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dinov3_vits16")
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    ap.add_argument("--out", type=Path, default=Path("artifacts/onnx"))
    ap.add_argument("--opset", type=int, default=13, help="TRT 8.2 parses 13 safely")
    ap.add_argument("--l2", dest="l2", action="store_true", default=True)
    ap.add_argument("--no-l2", dest="l2", action="store_false")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    # Export on CPU: tracing on MPS has bitten this repo before and the graph is device-free.
    model = PatchTokens(load_dinov3(args.name, device="cpu"), l2=args.l2).eval()

    check(model, sizes[0])
    if args.check_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    for h, w in sizes:
        tag = "" if args.l2 else "_raw"
        path = args.out / f"{args.name}_{h}x{w}{tag}.onnx"
        with opset13_attention(), frozen_rope(model.backbone, h // PATCH, w // PATCH):
            torch.onnx.export(
                model,
                torch.randn(1, 3, h, w),
                str(path),
                input_names=["image"],
                output_names=["feat"],
                opset_version=args.opset,
                dynamo=False,  # legacy tracer: static shapes, no onnxscript, TRT-friendly graph
                do_constant_folding=True,
            )
        import onnx

        m = onnx.load(str(path))
        if m.ir_version > TRT82_IR_VERSION:
            m.ir_version = TRT82_IR_VERSION
            onnx.save(m, str(path))
        onnx.checker.check_model(str(path))
        mb = path.stat().st_size / 1e6
        print(
            f"{path}  {h // PATCH}x{w // PATCH}={(h // PATCH) * (w // PATCH)} tokens  {mb:.1f} MB"
        )


if __name__ == "__main__":
    main()
