# Plan

Goal: show, with numbers on public data, that occlusion robustness in VOS comes from **how memory is written**, not from a bigger encoder. Frozen DINOv3 throughout.

## Milestones

| # | What | Done when |
|---|---|---|
| M0 | Scaffold, unit tests, config | `make test` green on a clean clone |
| M1 | Feature cache: DAVIS val+train, `clean` + `occ0` + `occ1` at 480×864, ViT-S/16 fp16 | `data/features/{val,train}` populated, ~6 GB |
| M2 | Zero-shot baseline on val, clean and occluded | `runs/baseline_*/summary.md` |
| M3 | Trained memory head, laptop budget (≤ 2 h on M3 Pro) | `runs/head_*/summary.md`, curve in `log.csv` |
| M4 | Ablations (below) | one table in README |
| M5 | Write-up: article in `docs/`, qualitative figure best/median/worst, "what this does NOT show" | README results section |

## Ablations, in order of expected signal

1. Gated vs ungated memory write (same head, `vis_gate` 0.5 vs 0.0) — the headline.
2. Permanent frame-0 entry vs pure FIFO.
3. Trained with occluded variants vs clean only — does the synthetic protocol transfer to the real episodes DAVIS already contains.
4. ViT-S/16 vs ViT-B/16 (both cached weights) — does the encoder matter once the memory is right.
5. Optional: LoRA on the last 4 blocks + Gram consistency loss — only if 1–4 leave headroom.

## Metrics reported

J, F, J&F on val (frames 1..T-1, DAVIS convention); `recovery_delay` median and never-recovered count over synthetic + real episodes; `leak_ratio` during episodes; `visibility_auc`.

## Compute

Extraction: ~50 ms/frame ViT-S at 480×864 on MPS → ~6 k frames × 3 variants ≈ 15 min. Training on cached features: head only, clip length 8, 400 clips/epoch, 15 epochs ≈ 40–60 min. No GPU cluster needed.

## Risks

- Synthetic occluders are pasted, not rendered: no shadows, no motion blur. Real-episode metrics guard against overfitting to the paste artefact.
- DAVIS real occlusions are rare and short; the never-recovered count will be small-sample.
- MPS `topk`/`scatter` on large (hw × N) tensors: verify memory at 30×54 × 8 frames before scaling resolution.
- Out-of-view vs occlusion are conflated in DAVIS annotations; reported as one class.

## Reading

DINOv3 (Siméoni et al. 2025, arXiv 2508.10104), DINO video-segmentation protocol (Caron et al. 2021), XMem (Cheng & Schwing 2022), Cutie (Cheng et al. 2024), SAM 2 memory attention (Ravi et al. 2024), Space-Time Correspondence as a Contrastive Random Walk (Jabri et al. 2020).
