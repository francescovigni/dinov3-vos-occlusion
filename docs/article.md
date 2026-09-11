# Tracking through occlusion is a memory-write problem

*Frozen DINOv3 features, a 3.4M-parameter memory head, DAVIS 2017, one laptop.*

## Question

Semi-supervised video object segmentation gives you a mask on frame 0 and asks for it on every later frame. Dense self-supervised features already do this zero-shot: copy labels from the most similar patches in recent frames. Where that recipe breaks is occlusion. The moment something passes in front of the object, the "most similar recent patches" are on the occluder, the propagated label follows them, and nothing brings it back when the object reappears.

The claim tested here: the fix is in **when memory is written**, not in the encoder. Three rules, none of them new, all of them cheap:

1. Do not write to memory while the object is not visible.
2. Keep frame 0 in memory permanently, so re-identification matches the clean template.
3. Predict visibility explicitly, so the tracker can say "hidden" instead of guessing.

The encoder stays frozen throughout. Every number below is "DINOv3 ViT-S/16 features + a small head".

## Setup

**Data.** DAVIS 2017, 480p. 60 training sequences, 30 validation sequences, the largest object on frame 0 of each sequence as the target (the first object id is a 0.0 % speck in `lab-coat` and a 0.3 % one in `scooter-black`). Nothing else.

**Synthetic occlusions with ground truth.** For each sequence, an object cut from a *training* sequence is pasted over the target for a contiguous episode of 4–12 frames, centred on the target's centroid with 5 % jitter, scaled 1.2–1.8× the target's box. The visible mask, full mask, occluder mask and occluded fraction are stored per frame, so recovery can be measured exactly. A frame is *hidden* when the occluded fraction is ≥ 0.9 or the mask is empty. Two seeds per validation sequence (`occ0`, `occ1`); `clean` is the untouched sequence.

The first version of this protocol used 0.8–1.4× occluders with 15 % jitter. It produced partial occlusions almost exclusively: 39 hidden frames out of 1,969 on validation and none in the held-out training sequences. A visibility head cannot learn from that, the gate never fired, and the ablations that depend on it came out identical. The numbers below are all on the corrected protocol.

**Real occlusions.** Any frame where the annotated object has zero area between two non-empty frames counts as a real episode. DAVIS has few, and they conflate occlusion with leaving the frame. They are reported alongside the synthetic ones.

**Features.** DINOv3 ViT-S/16 (21.6M parameters, frozen), last-layer patch tokens at 480×864 → 30×54×384, cached once in fp16. Validation at every frame with three variants; training at temporal stride 2 with `clean` and `occ0`, which is standard VOS practice (training samplers skip frames anyway) and keeps the cache at 12 GB instead of 23.

**Baseline.** k-NN label propagation as in the DINO video-segmentation protocol: for each patch at frame *t*, the 5 most similar patches within a 12-patch neighbourhood in frame 0 and the last 7 frames, softmax at temperature 0.07, weighted copy of their soft labels.

**Head.** Key (64-d) and value (256-d) projections of the features, the value conditioned on the mask at feature resolution; the key input also carries 32 fixed 2-D sinusoidal position channels. Memory = frame 0 (permanent) plus up to 7 later frames (FIFO), written as hard masks. Readout = softmax attention from the current frame's keys to memory keys, top-32 per query, restricted to a 12-patch neighbourhood for the later frames (frame 0 stays global). The same memory also yields the zero-shot k-NN label propagation, which enters the decoder as a prior channel and the visibility MLP as two pooled scalars. A three-block convolutional decoder upsamples to 1/4 resolution mask logits. Loss = BCE + Dice on the visible mask, BCE on visibility. Clips of 12 frames sampled with random gaps of 1–2 cached frames, teacher forcing on the memory write decaying from 1.0 to 0.2, AdamW 2e-4, cosine, 15 epochs of 300 clips. Eight training sequences are held out: their clean variant selects the checkpoint, their occluded variant calibrates the visibility gate. Trains in about 40 minutes on an M3 Pro.

**How the head got here.** Four versions, each fixing the failure the previous one exposed on validation:

| version | change | what it fixed | what it exposed |
|---|---|---|---|
| v1 | memory head as above, soft masks written to memory, fixed 0.5 gate | — | masks shrank frame by frame until the object vanished, on clean sequences too |
| v2 | hard masks written to memory, gapped clips, held-out checkpoint selection | the drift | collapses on scenes with several similar instances (fish, people) that the zero-shot k-NN survives because it is spatially local |
| v3 | position channels in the keys, locality window on recent memory | most of the instance confusion | still 5 points below zero-shot on clean; the gate never fired because the protocol had almost no fully hidden frames |
| v4 | zero-shot propagation fed to the head as a prior; heavier occluders; hidden = fraction ≥ 0.9; calibrated gate | — | see results |

**Metrics.** J (region IoU) and F (boundary F-measure) on frames 1..T−1, DAVIS convention, plus three occlusion numbers computed over the stored episodes: `recovery` (frames after the occluder leaves until J ≥ 0.5 again), `leak` (fraction of predicted mask sitting on the occluder while the object is hidden), and `vis AUC` (does the visibility score separate hidden from visible frames).

## Results

*(filled from `runs/results.md` once the pipeline has run)*

## What this does not show

- No fine-tuning of DINOv3. A LoRA ablation with a Gram-matrix anchor is the obvious next step and was not run.
- Single target per sequence, the largest on frame 0. Multi-object DAVIS scoring is not implemented.
- Pasted occluders have no shadows or motion blur. The real-episode numbers are the guard against learning the paste artefact.
- ViT-S only. ViT-B and ViT-L weights are cached and one config line away; they were not run.
- One seed per training run.

## What this would take on your data

A folder of frames, one mask on the first frame per object, and roughly a minute of GPU per thousand frames for feature extraction. The head is small enough to train on a laptop. The synthetic occlusion protocol needs no annotation beyond what the first-frame mask already gives.
