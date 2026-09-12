# Plan

Goal: show, with numbers on public data, that occlusion robustness in VOS comes from **how memory is written**, not from a bigger encoder. Frozen DINOv3 throughout.

## Milestones

| # | What | Done when |
|---|---|---|
| M0 ✅ | Scaffold, unit tests, config | `make test` green on a clean clone |
| M1 ✅ | Feature cache at 480×864, ViT-S/16 fp16 (1.24 MB/frame): val every frame × 3 variants (~7.4 GB), train stride 2 × `clean`,`occ0` (~5.2 GB) | `data/features/{val,train}` populated, ~13 GB |
| M2 ✅ | Zero-shot baseline on val, clean and occluded | `runs/baseline_*/summary.md` — J&F 0.771 / 0.678 |
| M3 ✅ | Trained memory head, laptop budget (≤ 1 h on M3 Pro per run) | `runs/head_*/summary.md`, held-out curve in `log.csv`, gate in `gate.json` |
| M4 ✅ | Ablations (below) | README `v4 ablations` table; gate ablation +1.4 J&F, FIFO ±0 |
| M5 ✅ | Write-up: article in `docs/`, qualitative figure best/median/worst, "what this does NOT show" | README results section, `docs/article.md` |

## Ablations, in order of expected signal

1. Gated vs ungated memory write (same head, calibrated gate vs 0.0) — the headline.
2. Permanent frame-0 entry vs pure FIFO.
3. Trained with occluded variants vs clean only — does the synthetic protocol transfer to the real episodes DAVIS already contains.
4. ViT-S/16 vs ViT-B/16 (both cached weights) — does the encoder matter once the memory is right.
5. Optional: LoRA on the last 4 blocks + Gram consistency loss — not run.

Outcome (11 Sep 2026): 1 gated > ungated by 1.4 J&F once the protocol has real hidden frames; 2 no effect with the locality window; 3 occlusion training buys leak/visibility, costs 1.6–2.8 J&F; 4 not run (disk); 5 not run.

## Metrics reported

J, F, J&F on val (frames 1..T-1, DAVIS convention); `recovery_delay` median and never-recovered count over synthetic + real episodes; `leak_ratio` during episodes; `visibility_auc`.

## Compute

Extraction: ~50 ms/frame ViT-S at 480×864 on MPS → ~10 k frame-variants ≈ 10 min. Train frames are cached at stride 2: standard VOS practice samples frames with gaps anyway, and the full cache would not fit the 19 GB free on this laptop. Training on cached features: head only, clip length 8, 400 clips/epoch, 15 epochs ≈ 40–60 min. No GPU cluster needed.

## Risks

- Synthetic occluders are pasted, not rendered: no shadows, no motion blur. Real-episode metrics guard against overfitting to the paste artefact.
- DAVIS real occlusions are rare and short; the never-recovered count will be small-sample.
- MPS `topk`/`scatter` on large (hw × N) tensors: verify memory at 30×54 × 8 frames before scaling resolution.
- Out-of-view vs occlusion are conflated in DAVIS annotations; reported as one class.

## Reading

DINOv3 (Siméoni et al. 2025, arXiv 2508.10104), DINO video-segmentation protocol (Caron et al. 2021), XMem (Cheng & Schwing 2022), Cutie (Cheng et al. 2024), SAM 2 memory attention (Ravi et al. 2024), Space-Time Correspondence as a Contrastive Random Walk (Jabri et al. 2020).

## Log of protocol decisions

- **Target = largest object on frame 0** (11 Sep). First object id is a 0.0 % speck in `lab-coat`.
- **Train cache at stride 2, clean + occ0 only** (11 Sep). Disk.
- **Occluders 1.2–1.8×, jitter 0.05, hidden = fraction ≥ 0.9 or empty** (11 Sep). The 0.8–1.4× protocol produced 39 hidden frames in 1,969; nothing to learn or gate on.
- **Gate calibrated on held-out training sequences** (11 Sep). A fixed 0.5 threshold never fired: the visibility head ranks correctly (AUC 0.97) but is biased by the class imbalance.
- **Metrics verified against the reference implementation** (11 Sep, late). J was identical; the first boundary-F implementation ran ~0.01 high (morphological gradient + ellipse kernel instead of the half-pixel boundary map + Euclidean disk). Re-implemented from the algorithm (the reference is GPL), verified to machine precision, last frame excluded as in the official protocol, all evaluations re-run.

## Study 2 — polyp tracking under occlusion in colonoscopy (started 12 Sep 2026)

Same head, same metrics, same pipeline; the DAVIS numbers become the control. What changes: real disappearance episodes exist in the annotations, the occluders are instruments, and the backbone will need adaptation.

| # | What | Done when |
|---|---|---|
| P0 | Data on disk with licences noted: LDPolypVideo (boxes, 160 videos), PolypGen positive sequences (masks), Kvasir-Instrument (occluder bank), Kvasir-SEG (stills for adaptation). SUN-SEG requested separately | `data/polyp/` populated, `scripts/download_polyp*.sh` |
| P1 ✅ | Adapters + real-episode statistics: how many polyps disappear and come back, for how long | below |
| P2 ✅ | Feature cache (480×560, ViT-S/16, uint8) + zero-shot propagation on both datasets | `runs/polyp_baseline_*` — J&F 0.137 box IoU, 0.440 mask J |
| P3 ✅ | Head v4 retrained on polyp sequences with instrument occluders, gate calibrated on held-out videos | `runs/polyp_head_*` — 0.208 / 0.196; leak on PolypGen 0.52 → 0.01 |
| P4 | LoRA on the last DINOv3 blocks with a Gram anchor, trained on public polyp stills only — **the experiment the P2/P3 numbers call for** | `runs/polyp_lora_*` |
| P5 ✅ | Write-up + overlay videos | README study-2 section, `docs/videos/polyp_*` |

Metrics specific to this study: box IoU on LDPolypVideo (J on filled boxes), mask J on PolypGen; re-acquisition delay on *real* episodes; false re-detection rate after the polyp leaves (does the tracker latch onto a fold).

Confidentiality: public data only, nothing from the NDA'd work, no client names.

### P1 — LDPolypVideo TrainValid, real disappearance episodes (12 Sep 2026)

100 videos, 24,789 annotated frames at 560×480, median 190 frames per video (27–1010). 15.5 % of frames carry no box; 4.3 % carry more than one (boxes have no identity, so the target track is: largest box on frame 0, then IoU association, nearest centre after a gap).

**90 disappearance episodes in 43 videos** (a run of box-less frames between two boxed frames): median 14 frames, p90 51, max 135; 24 episodes last 30+ frames. Occlusion by folds/instruments and leaving the field of view are indistinguishable in the annotation and are reported as one class. The synthetic instrument occluders from Kvasir-Instrument are kept as a second, controlled variant.

Disk budget: features at 480×560 are 0.8 MB/frame; both splits are cached at stride 2 with a per-video cap, `clean` + `occ0` only. The DAVIS caches were deleted to make room (regenerable in ~10 min).

### P2/P3 outcome (12 Sep 2026)

Frozen DINOv3 ViT-S/16 is the bottleneck on colonoscopy: zero-shot propagation 0.137 box-IoU J&F on the 60 test clips (0.77 on DAVIS), head 0.208. Elsewhere-J 0.11 vs 0.25. Polyps: median 2.1 % of frame, ≈ 4 patches short side. On PolypGen masks the head kills the leak (0.52 → 0.01) and separates hidden frames (AUC 0.74). Decision: no more head work on frozen features; P4 (LoRA + Gram anchor on Kvasir-SEG/PolypGen stills) is the next step.
