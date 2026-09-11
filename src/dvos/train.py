"""Train the gated memory head on cached features (backbone frozen, never loaded here).

Clips are sampled with random frame gaps (curriculum against drift), memory writes use hard
masks as at test time, and whole sequences are held out for validation: the checkpoint with
the best held-out J is ``model.pt``, the final one ``last.pt``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dvos.backbone import pick_device
from dvos.config import load_config
from dvos.metrics import hidden_frames, jaccard
from dvos.model import MemoryBank, MemoryVOS, bce_dice, build_model, to_feature_res, track


def list_clips(features_root: Path, split: str, variants: list[str] | None = None) -> list[Path]:
    dirs = sorted(p.parent for p in (features_root / split).glob("*/*/feats.npy"))
    if variants:
        dirs = [d for d in dirs if d.name in variants]
    return dirs


def sample_indices(n: int, length: int, gap_max: int, rng: random.Random) -> list[int]:
    """``length`` frame indices from a sequence of ``n`` with random gaps in [1, gap_max]."""
    length = min(length, n)
    gaps = [rng.randint(1, gap_max) for _ in range(length - 1)]
    span = sum(gaps)
    if span > n - 1:  # not enough room: fall back to consecutive frames
        gaps, span = [1] * (length - 1), length - 1
    start = rng.randint(0, n - 1 - span)
    idx = [start]
    for g in gaps:
        idx.append(idx[-1] + g)
    return idx


def load_clip(d: Path, idx: list[int], device: torch.device, hidden_thr: float = 0.9):
    """Features (T,C,h,w), visible masks (T,1,4h,4w) and hidden flags (T,) for the frames ``idx``."""
    feats = np.load(d / "feats.npy", mmap_mode="r")[idx]
    vis = np.load(d / "masks.npz")["visible"][idx]
    fraction = np.array(json.loads((d / "meta.json").read_text())["fraction"])[idx]
    hidden = torch.from_numpy(hidden_frames(vis, fraction, hidden_thr)).to(device)
    f = torch.from_numpy(np.array(feats, dtype=np.float32, copy=True)).to(device)
    m = torch.from_numpy(np.array(vis, dtype=np.float32, copy=True)).unsqueeze(1)  # (T,1,H,W)
    h, w = f.shape[2:]
    # area pooling to a non-divisible size is unsupported on MPS: resize on CPU, then move
    m = (F.interpolate(m, size=(4 * h, 4 * w), mode="area") > 0.5).float().to(device)
    return f, m, hidden


@torch.no_grad()
def validate(model: MemoryVOS, dirs: list[Path], mc, device: torch.device) -> float:
    """Mean J over full held-out sequences, tracked exactly as at test time."""
    model.eval()
    js = []
    for d in dirs:
        feats = torch.from_numpy(np.load(d / "feats.npy").astype(np.float32)).to(device)
        gt = np.load(d / "masks.npz")["visible"].astype(bool)
        h, w = feats.shape[2:]
        first = torch.from_numpy(gt[0].astype(np.float32))[None, None]
        first = (F.interpolate(first, size=(4 * h, 4 * w), mode="area") > 0.5).float().to(device)
        probs, _ = track(model, feats, first, memory_max=mc.memory_max, vis_gate=mc.vis_gate)
        for t in range(1, len(gt)):
            up = F.interpolate(probs[t], size=gt.shape[1:], mode="bilinear", align_corners=False)
            js.append(jaccard((up[0, 0] > 0.5).cpu().numpy(), gt[t]))
    model.train()
    return float(np.mean(js)) if js else float("nan")


def train_clip(model: MemoryVOS, feats, masks, hidden, mc, tf: float, rng: random.Random):
    """One clip: frame 0 from ground truth, then predict, accumulate losses, write memory."""
    length = feats.shape[0]
    h, w = feats.shape[2:]
    mem = MemoryBank(mc.memory_max)
    model.write(mem, feats[0:1], (to_feature_res(masks[0:1], (h, w)) > 0.5).float())
    lm = torch.zeros((), device=feats.device)
    lv = torch.zeros((), device=feats.device)
    for t in range(1, length):
        o = model.step(feats[t : t + 1], mem)
        target = masks[t : t + 1]
        vis_label = (~hidden[t]).float().view(1, 1)
        lm = lm + bce_dice(o["mask_logits"], target)
        lv = lv + F.binary_cross_entropy_with_logits(o["vis_logit"], vis_label)
        if rng.random() < tf:
            visible_now, mask_mem = bool(vis_label.item()), target
        else:
            visible_now = torch.sigmoid(o["vis_logit"]).item() > mc.vis_gate
            mask_mem = (torch.sigmoid(o["mask_logits"]) > 0.5).float().detach()
        if visible_now:
            model.write(mem, feats[t : t + 1], (to_feature_res(mask_mem, (h, w)) > 0.5).float())
    n = max(1, length - 1)
    return lm / n, lv / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="runs/head")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--clips-per-epoch", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=None, help="train only on these variants")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tr, mc = cfg.train, cfg.model
    epochs = args.epochs or tr.epochs
    clips_per_epoch = args.clips_per_epoch or tr.clips_per_epoch
    torch.manual_seed(tr.seed)
    rng = random.Random(tr.seed)
    device = pick_device(cfg.backbone.device)

    clips = list_clips(Path(cfg.data.features_root), args.split, args.variants)
    if not clips:
        raise SystemExit("no cached features; run dvos.extract first")
    # hold out whole sequences (all their variants); validate on their clean variant
    seqs = sorted({d.parent.name for d in clips})
    holdout = set(seqs[-tr.holdout :]) if tr.holdout > 0 else set()
    val_dirs = [d for d in clips if d.parent.name in holdout and d.name == "clean"]
    clips = [d for d in clips if d.parent.name not in holdout]
    if not clips:
        raise SystemExit("holdout leaves no training sequences")

    c_in = int(np.load(clips[0] / "feats.npy", mmap_mode="r").shape[1])
    model = build_model(c_in, mc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * clips_per_epoch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(dict(holdout=sorted(holdout), train=seqs), indent=1))
    best = -1.0
    with open(out / "log.csv", "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "loss", "loss_mask", "loss_vis", "teacher_forcing", "val_J"])
        for epoch in range(epochs):
            tf = tr.teacher_forcing_start + (tr.teacher_forcing_end - tr.teacher_forcing_start) * (
                epoch / max(1, epochs - 1)
            )
            sums = np.zeros(3)
            model.train()
            for _ in tqdm(range(clips_per_epoch), desc=f"epoch {epoch}", leave=False):
                d = rng.choice(clips)
                n = np.load(d / "feats.npy", mmap_mode="r").shape[0]
                idx = sample_indices(n, tr.clip_len, tr.gap_max, rng)
                feats, masks, hidden = load_clip(d, idx, device, cfg.occlusion.hidden_fraction)
                lm, lv = train_clip(model, feats, masks, hidden, mc, tf, rng)
                loss = tr.w_mask * lm + tr.w_vis * lv
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                sums += np.array([loss.item(), lm.item(), lv.item()])
            avg = sums / clips_per_epoch
            val_j = validate(model, val_dirs, mc, device) if val_dirs else float("nan")
            writer.writerow([epoch, *[f"{x:.4f}" for x in avg], f"{tf:.2f}", f"{val_j:.4f}"])
            log.flush()
            print(
                f"epoch {epoch}: loss {avg[0]:.4f} mask {avg[1]:.4f} vis {avg[2]:.4f} "
                f"tf {tf:.2f} val_J {val_j:.4f}"
            )
            state = dict(
                model=model.state_dict(),
                c_in=c_in,
                epoch=epoch,
                val_J=val_j,
                config=json.loads(json.dumps(cfg, default=vars)),
            )
            torch.save(state, out / "last.pt")
            if not val_dirs or val_j > best:
                best = val_j
                torch.save(state, out / "model.pt")


if __name__ == "__main__":
    main()
