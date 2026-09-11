"""Train the gated memory head on cached features (backbone frozen, never loaded here)."""

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
from dvos.model import MemoryBank, MemoryVOS, bce_dice, to_feature_res


def list_clips(features_root: Path, split: str, variants: list[str] | None = None) -> list[Path]:
    dirs = sorted(p.parent for p in (features_root / split).glob("*/*/feats.npy"))
    if variants:
        dirs = [d for d in dirs if d.name in variants]
    return dirs


def load_clip(d: Path, start: int, length: int, device: torch.device):
    feats = np.load(d / "feats.npy", mmap_mode="r")[start : start + length]
    masks = np.load(d / "masks.npz")
    vis = masks["visible"][start : start + length]
    f = torch.from_numpy(np.array(feats, dtype=np.float32, copy=True)).to(device)
    m = (
        torch.from_numpy(np.array(vis, dtype=np.float32, copy=True)).unsqueeze(1).to(device)
    )  # (T,1,H,W)
    h, w = f.shape[2:]
    m = (F.interpolate(m, size=(4 * h, 4 * w), mode="area") > 0.5).float()
    return f, m


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
    random.seed(tr.seed)
    device = pick_device(cfg.backbone.device)
    clips = list_clips(Path(cfg.data.features_root), args.split, args.variants)
    if not clips:
        raise SystemExit("no cached features; run dvos.extract first")
    c_in = int(np.load(clips[0] / "feats.npy", mmap_mode="r").shape[1])
    model = MemoryVOS(c_in, mc.c_key, mc.c_value, mc.hidden, mc.readout_topk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * clips_per_epoch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "log.csv", "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "loss", "loss_mask", "loss_vis", "teacher_forcing"])
        step = 0
        for epoch in range(epochs):
            tf = tr.teacher_forcing_start + (
                tr.teacher_forcing_end - tr.teacher_forcing_start
            ) * epoch / max(1, epochs - 1)
            sums = np.zeros(3)
            model.train()
            for _ in tqdm(range(clips_per_epoch), desc=f"epoch {epoch}", leave=False):
                d = random.choice(clips)
                n = np.load(d / "feats.npy", mmap_mode="r").shape[0]
                length = min(tr.clip_len, n)
                start = random.randint(0, n - length)
                feats, masks = load_clip(d, start, length, device)
                h, w = feats.shape[2:]
                mem = MemoryBank(mc.memory_max)
                mem.add(*model.encode(feats[0:1], to_feature_res(masks[0:1], (h, w))))
                lm = torch.zeros((), device=device)
                lv = torch.zeros((), device=device)
                for t in range(1, length):
                    mk, mv = mem.read()
                    o = model(feats[t : t + 1], mk, mv)
                    target = masks[t : t + 1]
                    vis_label = (target.sum() > 0).float().view(1, 1)
                    lm = lm + bce_dice(o["mask_logits"], target)
                    lv = lv + F.binary_cross_entropy_with_logits(o["vis_logit"], vis_label)
                    use_gt = random.random() < tf
                    if use_gt:
                        visible_now, mask_mem = bool(vis_label.item()), target
                    else:
                        visible_now = torch.sigmoid(o["vis_logit"]).item() > mc.vis_gate
                        mask_mem = torch.sigmoid(o["mask_logits"]).detach()
                    if visible_now:
                        mem.add(*model.encode(feats[t : t + 1], to_feature_res(mask_mem, (h, w))))
                lm, lv = lm / max(1, length - 1), lv / max(1, length - 1)
                loss = tr.w_mask * lm + tr.w_vis * lv
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                sums += np.array([loss.item(), lm.item(), lv.item()])
            avg = sums / clips_per_epoch
            writer.writerow([epoch, *[f"{x:.4f}" for x in avg], f"{tf:.2f}"])
            log.flush()
            print(
                f"epoch {epoch}: loss {avg[0]:.4f} mask {avg[1]:.4f} vis {avg[2]:.4f} tf {tf:.2f}"
            )
            torch.save(
                dict(
                    model=model.state_dict(),
                    c_in=c_in,
                    config=json.loads(json.dumps(cfg, default=vars)),
                ),
                out / "model.pt",
            )


if __name__ == "__main__":
    main()
