"""DINOv3 loading and dense feature extraction.

Model code comes from a local clone of facebookresearch/dinov3 (sibling ``../dinov3`` by
default, or ``$DINOV3_REPO``). Weights come from ``~/.cache/torch/hub/checkpoints`` where
``torch.hub`` put them after the licence was accepted once. Nothing here downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 16
HUB_CACHE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
EMBED_DIM = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
}


def default_repo() -> Path:
    env = os.environ.get("DINOV3_REPO")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "dinov3"


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_cached_weights(name: str) -> Path | None:
    hits = sorted(HUB_CACHE.glob(f"{name}_pretrain_lvd1689m*.pth"))
    return hits[0] if hits else None


def _build_backbone(repo: Path, name: str) -> torch.nn.Module:
    """Import ``dinov3.hub.backbones.<name>`` from the local clone without touching hubconf.py.

    hubconf imports the segmentation/depth heads too, which need torchmetrics and friends;
    the backbones module only needs torch.
    """
    import importlib
    import sys

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    backbones = importlib.import_module("dinov3.hub.backbones")
    if not hasattr(backbones, name):
        raise ValueError(f"unknown backbone {name!r}; see dinov3/hub/backbones.py")
    return getattr(backbones, name)(pretrained=False)


def load_dinov3(
    name: str = "dinov3_vits16",
    repo: str | Path | None = None,
    weights: str | Path | None = None,
    device: str | torch.device = "auto",
) -> torch.nn.Module:
    """Frozen DINOv3 backbone in eval mode on ``device``."""
    repo = Path(repo) if repo else default_repo()
    if not (repo / "hubconf.py").exists():
        raise FileNotFoundError(f"DINOv3 repo not found at {repo}; set DINOV3_REPO")
    weights = Path(weights) if weights else find_cached_weights(name)
    if weights is None or not weights.exists():
        raise FileNotFoundError(
            f"no cached weights for {name} under {HUB_CACHE}; accept the DINOv3 licence, "
            "download the checkpoint, or pass backbone.weights in the config"
        )
    model = _build_backbone(repo, name)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    dev = pick_device(device) if isinstance(device, str) else device
    return model.to(dev)


def preprocess(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """uint8 HxWx3 -> normalised 1x3xHxW at ``size`` (both multiples of 16)."""
    h, w = size
    if h % PATCH or w % PATCH:
        raise ValueError(f"size {size} must be multiples of {PATCH}")
    x = (
        torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
    )
    x = F.interpolate(x[None], size=(h, w), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def extract_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Bx3xHxW normalised -> BxCxH/16xW/16 patch tokens, last layer, layer-normed."""
    device = next(model.parameters()).device
    feats = model.get_intermediate_layers(x.to(device), n=1, reshape=True, norm=True)[0]
    return feats.float()
