# Foundation models, measured honestly, on real hardware

> I took a frozen DINOv3 video-segmentation stack from a workstation to a €100 Jetson Nano with a camera on it, and measured what actually happened. Three findings came out, none of them the latency number I went in for. All code, data and failures are here.

**The 60-second version**

- **The fast engine computed nothing — and the standard benchmark tool certified it as healthy.** Every FP16 TensorRT engine returned NaN for every pixel. `trtexec` reported clean throughput for all of them, because it feeds random input and never reads the output.
- **The benchmark was blind to the failure that mattered.** Dropping resolution to buy frame rate costs 3.4 J&F points on DAVIS — and the entire track on real video. The number that governs the decision cannot see the thing the decision breaks.
- **Diagnosing that inverted the trade-off.** A re-detection branch made the *cheap* configuration the better one: 2× the frame rate **and** better post-occlusion tracking.
- **The headline model never beat its own baseline.** Four trained memory heads, 3.4M parameters each, none better than zero-shot k-NN on J&F. I am showing you anyway — the diagnosis is the deliverable, and it says *encoder*, not *method*.

![accuracy against frame rate on the Jetson Nano; every point left of the real-time line](docs/figures/edge_pareto.png)

Full quality runs at **0.75 fps — 33× short of real time**. There is no operating point on this curve where this architecture is live on this board. Saying so is the result.

---

## 1. The fast engine computed nothing

The engines built. They ran. `trtexec` reported 9.04 fps at 192×336 with tight percentiles, and an independently written runner reproduced those timings on real camera frames to within 1%. Two agreeing measurements.

Then the saved features — 113 MB of them — gzipped to 153 KB, because they were uniform NaN.

```
fp32 feat_0000.npy  finite 96768/96768  min -0.2223  max 0.4502
fp16 feat_0000.npy  finite      0/96768  min     nan  max     nan
```

TensorRT 8.2 has no native LayerNormalization op, so the exported graph carries the decomposition, and the variance term `Pow(x, 2)` overflows fp16's 65504 ceiling on ViT activations. Numerically dead from the first block. Correctness costs ~1.4× latency.

**A throughput number from `trtexec` certifies that an engine runs, not that it computes the function.** A large fraction of published edge-inference numbers are produced exactly this way, and nothing in the tool's output distinguishes the two cases. The three-line finiteness check in `jetson_infer.py` is the only reason this is a finding rather than a published number.

## 2. The benchmark was blind to the failure that mattered

![DAVIS sees a 3.4-point drop; the real camera scene loses the track entirely](docs/figures/resolution_cliff.png)

At 384×672 the mask tracks the object, loses it to a real hand occlusion, and never re-acquires it — through a full reappearance and a second occlusion. Re-acquisition goes 2 of 2 → 0 of 2; the visibility head goes 0.845 AUC → 0.534, which is chance. DAVIS cannot see any of it, because DAVIS val contains few full occlusions and scores an average over frames rather than asking whether the object was ever found again.

**The obvious way to buy frame rate on an edge device silently removes the one capability the system exists for.**

Getting ground truth for this cost no labelling. The trick is in the scene, not the code: make the *target* trivially segmentable (an orange ball, a hue window) and let the *occluder* be arbitrary (a hand). Exact per-frame masks on real optics with real auto-exposure, and the target is still free to move.

## 3. The fix inverted the trade-off

A global cosine match between the frame-0 target and every patch of a later frame separates cleanly at 384×672, long after the mask has died — on-target 0.95 against background p99 0.68. **The encoder never lost the object; the voting scheme did.** So use the exemplar *outside* the vote: when the mask dies, stop propagating and re-seed from a global match.

| configuration | fps | J after occlusion |
|---|---|---|
| 480×864, propagation only | 0.75 | 0.646 |
| **384×672 + gated re-detection** | **1.50** | **0.668** |

Twice the frame rate and slightly better post-occlusion tracking. The make-or-buy answer is not "lower resolution costs you the track" — it is "lower resolution costs you the track *unless* you add the branch, and with it the frame rate is free."

Full write-up, including two wrong guesses and the measurement that ended them: **[docs/edge.md](docs/edge.md)**.

---

## What the system is

Semi-supervised video object segmentation: given a mask on frame 0, propagate it through video with frozen DINOv3 features and a small trainable memory head that survives occlusion — stops updating under hidden state, signals visibility, re-acquires on reappearance.

```mermaid
flowchart LR
    A[Frame t] --> B[DINOv3 ViT-S/16<br/>frozen, patch tokens 1/16]
    B --> C[Key / Value<br/>projections]
    M[(Memory bank<br/>frame 0 permanent + last N)] --> D[Attention readout]
    C --> D
    D --> E[Mask decoder<br/>1/4 res logits]
    D --> V[Visibility head<br/>hidden / visible]
    V -->|visible| M
    E --> F[Mask t]
    Z[Zero-shot baseline:<br/>k-NN label propagation] -.->|same features| F
```

k-NN label propagation on DINOv3 works until something occludes the target: the memory absorbs the occluder and the mask drifts onto it. Three fixes — **gated memory update** (freeze memory when visibility says hidden), **visibility output** (trained on frames where ground truth is empty), **permanent frame-0 memory** (re-identify against a clean template after reappearance). Backbone stays frozen throughout.

**Why standard metrics were not enough.** J&F says the tracker is fine while it paints the occluder. So three more, all computed from stored episodes: `leak_ratio` (how much of the mask sits on the occluder while the object is hidden), `recovery_delay` (frames until J ≥ 0.5 after the occluder leaves), `visibility_auc` (does the model know it cannot see). J and F are re-implemented here under MIT and checked against the GPL reference `davis2017-evaluation`: J identical, F identical to machine precision (`tests/test_metrics.py`).

![baseline left, head right — the baseline paints the occluder, the head returns empty and says HIDDEN](docs/videos/dog_occ0.gif)

Left: zero-shot baseline paints the occluder (J = 0, still "visible"). Right: the head returns an empty mask (J = 1, "hidden"). Both re-acquire three frames later.

## The honest scoreboard — DAVIS 2017 val

**None of the four heads beat zero-shot on J&F.** Training with occlusions does not buy accuracy, it buys *behaviour*: leak 0.93 → 0.60, visibility AUC 0.80 → 0.91, at a cost of 4–7 J&F points elsewhere. Reproduce with `scripts/run_all.sh` then `scripts/run_gated.sh`; tables generated by `scripts/render_results.py --inject README.md`.

<!-- results:start -->
#### Main comparison (DAVIS 2017 val, 30 sequences, largest object on frame 0)

| method | clean J&F | occ0 J&F | occ1 J&F | occ0 leak | occ0 vis AUC | occ0 never recovered |
|---|---|---|---|---|---|---|
| zero-shot k-NN propagation | 0.767 | 0.673 | 0.682 | 0.927 | 0.797 | 3 / 30 |
| zero-shot propagation + learned visibility gate | 0.767 | 0.673 | 0.682 | 0.927 | 0.848 | 3 / 30 |
| head v1 (soft memory masks) | 0.637 | 0.589 | 0.597 | 0.466 | 0.753 | 6 / 30 |
| head v2 (+ hard masks, gapped clips, held-out selection) | 0.670 | 0.630 | 0.617 | 0.544 | 0.733 | 4 / 30 |
| head v3 (+ position channels, locality window) | 0.708 | 0.633 | 0.631 | 0.674 | 0.643 | 5 / 30 |
| head v4 (+ zero-shot prior, calibrated gate) | 0.702 | 0.625 | 0.629 | 0.596 | 0.908 | 3 / 30 |

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
| v4, calibrated gate, frame 0 permanent | 0.625 | 0.596 | 0.908 | 3 / 30 |
| v4, gate chosen by J | 0.611 | 0.566 | 0.848 | 4 / 30 |
| v4, ungated writes | 0.611 | 0.566 | 0.848 | 4 / 30 |
| v4, FIFO memory (frame 0 evictable) | 0.622 | 0.594 | 0.907 | 3 / 30 |
| v4 trained on clean features only | 0.642 | 0.878 | 0.632 | 2 / 30 |
| v4 trained on clean only, evaluated on clean | 0.731 | – | – | – |
<!-- results:end -->

**How to read it.** Zero-shot loses 9 J&F under occlusion, all of it inside the episode: it paints the occluder (leak 0.93, J 0.05 inside) and snaps back the moment the occluder leaves, because frame 0 stays in context. Its weakness is not re-acquisition — it is not knowing the object is gone. Gated writes beat ungated by +1.4 J&F and +0.06 visibility AUC. Applying the head's visibility score as a hard gate on zero-shot does not survive calibration; an oracle gate of 0.3 would give +1.5 J and halve the leak, which is the size of the prize if the gate were better. Qualitative strips, per-sequence breakdowns and the full argument: **[docs/article.md](docs/article.md)**.

## Does it transfer? Colonoscopy says: not on frozen features

Same head, same metrics, same pipeline, on public colonoscopy video where the object genuinely disappears — behind folds, behind instruments, out of the field of view. [LDPolypVideo](https://github.com/dashishi/LDPolypVideo-Benchmark) (160 clips, boxes), [PolypGen](https://github.com/DebeshJha/PolypGen) (masks), [Kvasir-Instrument](https://datasets.simula.no/kvasir-instrument/) (occluder bank).

<!-- polyp-results:start -->
**Frozen DINOv3 features do not separate polyps from mucosa.** Zero-shot reaches J&F 0.137 on LDPolypVideo test against 0.767 on DAVIS. The memory head beats it by 7 points (0.208) and re-acquires twice as fast after real disappearances (median 4.5 frames vs 9), but from weak signal.

| method | clean J&F | occ0 J&F | occ0 leak | occ0 vis AUC | real episodes never re-acquired |
|---|---|---|---|---|---|
| zero-shot k-NN propagation | 0.137 | 0.136 | 0.078 | 0.642 | 21 / 38 |
| head, gated (calibrated gate 0.4) | 0.208 | 0.196 | 0.121 | 0.696 | 19 / 38 |

The visibility head is the part that transfers: on mask-level PolypGen, leak drops **0.52 → 0.01** and visibility AUC rises **0.48 → 0.74**. Polyps are small — median 2.1% of frame, short side ≈ 4 patches — and zero-shot locks onto mucosa texture.

**The numbers say encoder, not method.** Next experiment is LoRA on the last DINOv3 blocks with a Gram anchor, not another head. One run, one seed, boxes with heuristic matching.

![best occluded colonoscopy clip; zero-shot baseline left, memory head right](docs/videos/polyp_133_occ0.gif)
<!-- polyp-results:end -->

## What this does NOT show

- **No fine-tuning of DINOv3.** Every number is frozen features + a small head. The LoRA ablation is planned, not done.
- **One board, one backbone, one camera scene.** The Jetson Nano is 2019 silicon with no tensor cores; none of the FP16 behaviour should be assumed to transfer to an Orin. The NaN is specific to TensorRT 8.2 — ≥ 8.6 has a native LayerNorm. The transferable claim is about `trtexec` not validating outputs.
- **The resolution cliff is measured, not established.** 120 frames, two episodes, one object class. It is a reason to test re-acquisition on *your* footage before choosing a resolution, not a number to quote.
- **One run, one seed per configuration.** Point-or-two differences may be seed variance. Gate thresholds calibrated on held-out training sequences, never on val; checkpoint selection sees no val frame.
- **Single target only**, largest object on frame 0. Multi-object DAVIS not implemented, and head failures concentrate in multi-instance scenes.
- **No power or thermal measurement**, no INT8, no pruning, no distillation, no LayerNorm plugin. Each could move the Pareto; none were tried.

## What this would take on your data

- **A day** to port a frozen encoder and produce the latency/accuracy curve for your resolutions and your board — including the engine-build failures, which is where the surprises live.
- **A day** to build an automatic-ground-truth scene for your failure mode. The trick generalises: make the thing you are measuring trivially detectable and let whatever causes the failure stay arbitrary. That is what makes the numbers exact without anyone labelling a frame.
- **The deliverable** is the make-or-buy answer: the resolution and precision where your accuracy requirement and your frame-rate requirement actually intersect — and an honest statement when they do not intersect at all, which on this board for this model, they do not.

## Run it

```bash
make setup     # uv venv (py3.12) + deps
make test      # unit tests, no data, no weights needed
make data      # DAVIS 2017 trainval 480p (~800 MB)
make extract   # cache DINOv3 features: clean + 2 occluded variants
make baseline  # zero-shot propagation on val
make train     # memory head, laptop-scale
make eval      # checkpoint on val -> metrics JSON + tables
```

```
src/dvos/     backbone (frozen DINOv3) · davis · occlusion · propagate · model · metrics
              extract / train / evaluate / video  (CLIs)
tools/        ONNX export · Jetson capture & inference · scene eval · silhouette tracking · figures
scripts/      dataset downloads · run_all.sh · run_gated.sh · run_polyp.sh · render_results.py
docs/         edge.md (the Jetson write-up) · article.md (the DAVIS study) · figures · videos
```

Weights: `backbone.py` looks for DINOv3 checkpoints in `~/.cache/torch/hub/checkpoints/` and imports model code from a sibling `../dinov3` clone (override with `DINOV3_REPO`). Data directories are not in this repo — `make data` and `scripts/download_polyp.sh` fetch them.

## Licence

Code MIT. DINOv3 weights under Meta's DINOv3 licence, not MIT — accept it on the official page before downloading. DAVIS 2017 under CC BY 4.0; cite Pont-Tuset et al., *The 2017 DAVIS Challenge on Video Object Segmentation*, arXiv:1704.00675.
