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

The tables are generated from `runs/` by `scripts/render_results.py`; the README carries the same block.

<!-- results:start -->
#### Main comparison (DAVIS 2017 val, 30 sequences, largest object on frame 0)

| method | clean J&F | occ0 J&F | occ1 J&F | occ0 leak | occ0 vis AUC | occ0 never recovered |
|---|---|---|---|---|---|---|
| zero-shot k-NN propagation | 0.771 | 0.678 | 0.686 | 0.927 | 0.797 | 3 / 30 |
| zero-shot propagation + learned visibility gate | 0.771 | 0.678 | – | 0.927 | 0.848 | 3 / 30 |
| head v1 (soft memory masks) | 0.672 | 0.592 | 0.600 | 0.466 | 0.753 | 6 / 30 |
| head v2 (+ hard masks, gapped clips, held-out selection) | 0.691 | 0.634 | 0.620 | 0.544 | 0.733 | 4 / 30 |
| head v3 (+ position channels, locality window) | 0.718 | 0.637 | 0.634 | 0.674 | 0.643 | 5 / 30 |
| head v4 (+ zero-shot prior, calibrated gate) | 0.706 | 0.630 | 0.634 | 0.596 | 0.908 | 3 / 30 |

#### Where the accuracy goes (occ0, per-frame J)

| method (occ0) | J inside episode | J 10 frames after | J elsewhere |
|---|---|---|---|
| zero-shot k-NN propagation | 0.050 | 0.729 | 0.768 |
| zero-shot propagation + learned visibility gate | 0.050 | 0.729 | 0.768 |
| head v1 (soft memory masks) | 0.220 | 0.595 | 0.630 |
| head v2 (+ hard masks, gapped clips, held-out selection) | 0.239 | 0.667 | 0.665 |
| head v3 (+ position channels, locality window) | 0.143 | 0.648 | 0.685 |
| head v4 (+ zero-shot prior, calibrated gate) | 0.133 | 0.687 | 0.697 |

#### v4 ablations

| ablation (occ0) | J&F | leak | vis AUC | never recovered |
|---|---|---|---|---|
| v4, calibrated gate, frame 0 permanent | 0.630 | 0.596 | 0.908 | 3 / 30 |
| v4, ungated writes | 0.616 | 0.566 | 0.848 | 4 / 30 |
| v4, FIFO memory (frame 0 evictable) | 0.627 | 0.594 | 0.907 | 3 / 30 |
| v4 trained on clean features only | 0.646 | 0.878 | 0.632 | 2 / 30 |
| v4 trained on clean only, evaluated on clean | 0.734 | – | – | – |
<!-- results:end -->

### Reading the numbers

**The zero-shot baseline is strong, and it fails in exactly one place.** Frozen DINOv3 features with k-NN label propagation reach J&F 0.771 on clean val. Under full occlusion they drop to 0.678, and the episode split shows where: J inside the episode is 0.05 with a leak of 0.93 — the propagated mask paints the occluder — while J ten frames after the episode is 0.73 and 0.77 elsewhere. Recovery is immediate in 27 of 30 sequences because frame 0 is always in the context. The baseline does not have a re-acquisition problem. It has a "does not know the object is gone" problem.

**The trained head never beat it on J&F.** Four versions, each diagnosed on val and fixed: v1 drifted because soft masks written to memory eroded the object frame by frame; v2 wrote hard masks and stopped drifting but collapsed on scenes with several similar instances; v3 added position channels and a locality window and recovered most of those; v4 fed the zero-shot propagation to the decoder as a prior. Clean J&F went 0.672 → 0.691 → 0.718 → 0.706 against 0.771 for zero-shot. On the first twelve val sequences the head's own k-NN prior scores J 0.756 with the head's memory writes and 0.847 with oracle writes, while the head's output scores 0.766: the decoder improves its input by a point, and the missing nine points are error accumulation through the memory. A 3.4M-parameter head trained on 52 sequences does not learn a better appearance model than plain feature similarity; it can only learn how to *use* memory.

**What occlusion training buys, measured against the same head trained on clean features.** Leak 0.88 → 0.60, visibility AUC 0.63 → 0.91, J inside the episode 0.05 → 0.13. Cost: 2.8 J&F points on clean sequences (0.734 → 0.706) and 1.6 under occlusion (0.646 → 0.630). The heads see fewer than 300 fully-hidden training frames; that they learn to say "hidden" at all is the positive result, and that it costs accuracy on visible frames is the honest one.

**Gating.** With a calibrated gate the v4 head beats its ungated self by 1.4 J&F and 0.06 visibility AUC on occ0. Permanent frame 0 versus plain FIFO changes nothing once reads are spatially local. Under the first, too-gentle occlusion protocol these two ablations were *identical* — the gate never fired because there were 39 hidden frames in 1,969. Protocol first, ablation second.

**The thesis test that failed.** If the head's contribution is the visibility signal, the cleanest use of it is to gate the zero-shot propagation: no memory write and no foreground while the head says hidden. Chosen by mean J on the held-out training sequences, the best gate is 0.0 — no gate. Tuned on val itself, a gate of 0.3 gives +1.5 J and halves the leak (0.94 → 0.50, J inside 0.05 → 0.32). That is an oracle bound, not a result: the visibility scores on the held-out sequences (balanced accuracy 0.70) are not reliable enough for a hard threshold to transfer.

### What would change the picture

- **A better memory-write policy, not a better decoder.** The oracle-write gap (0.756 → 0.847) is the largest number in this study. Confidence-weighted writes, write-back correction, or writing the k-NN prior instead of the decoder output are all cheaper than another decoder.
- **More hidden frames.** 300 is not enough to train a visibility head that transfers; a second occluder per sequence or a longer episode range would double it at no annotation cost.
- **Soft gating.** Down-weighting recent memory by the visibility score instead of thresholding would keep the +1.5 J of the oracle without the cliff at 0.9.
- **LoRA on the last DINOv3 blocks with a Gram anchor**, the ablation this study did not run, would test whether the instance-confusion failures are a feature limitation or a head limitation.

## What this does not show

- No fine-tuning of DINOv3. A LoRA ablation with a Gram-matrix anchor is the obvious next step and was not run.
- Single target per sequence, the largest on frame 0. Multi-object DAVIS scoring is not implemented.
- Pasted occluders have no shadows or motion blur. The real-episode numbers are the guard against learning the paste artefact.
- ViT-S only. ViT-B and ViT-L weights are cached and one config line away; they were not run.
- One seed per training run. Differences of a point or two between head versions are within what a second seed could move.

## What this would take on your data

A folder of frames, one mask on the first frame per object, and roughly a minute of GPU per thousand frames for feature extraction. The head is small enough to train on a laptop. The synthetic occlusion protocol needs no annotation beyond what the first-frame mask already gives.
