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

**Data.** DAVIS 2017, 480p. 60 training sequences, 30 validation sequences, first object of each sequence as the target. Nothing else.

**Synthetic occlusions with ground truth.** For each sequence, an object cut from a *training* sequence is pasted over the target for a contiguous episode of 4–12 frames, centred on the target's centroid with jitter, scaled 0.8–1.4× the target's box. The visible mask, full mask, occluder mask and occluded fraction are stored per frame, so recovery can be measured exactly. Two seeds per validation sequence (`occ0`, `occ1`); `clean` is the untouched sequence.

**Real occlusions.** Any frame where the annotated object has zero area between two non-empty frames counts as a real episode. DAVIS has few, and they conflate occlusion with leaving the frame. They are reported alongside the synthetic ones.

**Features.** DINOv3 ViT-S/16 (21.6M parameters, frozen), last-layer patch tokens at 480×864 → 30×54×384, cached once in fp16. Validation at every frame; training at temporal stride 2, which is standard VOS practice (training samplers skip frames anyway) and halves the cache.

**Baseline.** k-NN label propagation as in the DINO video-segmentation protocol: for each patch at frame *t*, the 5 most similar patches within a 12-patch neighbourhood in frame 0 and the last 7 frames, softmax at temperature 0.07, weighted copy of their soft labels.

**Head.** Key (64-d) and value (256-d) projections of the features, the value conditioned on the mask at feature resolution. Memory = frame 0 (permanent) plus up to 7 later frames (FIFO). Readout = softmax attention from the current frame's keys to memory keys, top-32 per query. A three-block convolutional decoder upsamples the readout to 1/4 resolution mask logits; a pooled readout feeds a visibility MLP. Loss = BCE + Dice on the visible mask, BCE on visibility. Clips of 8 frames, teacher forcing on the memory write decaying from 1.0 to 0.2, AdamW 2e-4, cosine, 15 epochs of 400 clips. Trains in under an hour on an M3 Pro.

**Metrics.** J (region IoU) and F (boundary F-measure) on frames 1..T−1, DAVIS convention, plus three occlusion numbers computed over the stored episodes: `recovery` (frames after the occluder leaves until J ≥ 0.5 again), `leak` (fraction of predicted mask sitting on the occluder while the object is hidden), and `vis AUC` (does the visibility score separate hidden from visible frames).

## Results

*(filled from `runs/results.md` once the pipeline has run)*

## What this does not show

- No fine-tuning of DINOv3. A LoRA ablation with a Gram-matrix anchor is the obvious next step and was not run.
- Single target per sequence. Multi-object DAVIS scoring is not implemented.
- Pasted occluders have no shadows or motion blur. The real-episode numbers are the guard against learning the paste artefact.
- ViT-S only. ViT-B and ViT-L weights are cached and one config line away; they were not run.
- One seed per training run.

## What this would take on your data

A folder of frames, one mask on the first frame per object, and roughly a minute of GPU per thousand frames for feature extraction. The head is small enough to train on a laptop. The synthetic occlusion protocol needs no annotation beyond what the first-frame mask already gives.
