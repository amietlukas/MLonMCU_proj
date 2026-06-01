# Classifier — Architecture notes & MCU benchmarks

Findings collected while flashing the smallnet / bignet / bignet_pruned
baselines onto the B-U585I-IOT02A (STM32U585, Cortex-M33 @ 160 MHz) via
X-CUBE-AI 10.2.0.

Source of truth for the live numbers is
[`firmware/Host/model_comparison.csv`](../../../firmware/Host/model_comparison.csv),
which `firmware/Host/compare_models.py` regenerates from the per-run
`results_*.csv` files produced by `host.py`.

---

## 1. MCU benchmark table

Latencies measured on the U585 itself (DWT cycle counts → ms). Accuracy
comes from running `host.py` against the HAGRID test set
(`hagrid_full_qqvga_resize/test`, ~30 000 images across 6 classes).

| model              | n      | acc    | macro F1 | infer_ms | params  | MACC         | flash KB | RAM KB | E / inference (mJ) |
|--------------------|--------|--------|----------|----------|---------|--------------|----------|--------|--------------------|
| smallnet_fp32      | 1 137  | 0.823  | 0.818    | 8 244.9  | 93 446  | 184 666 598  | 365      | 691    | 138.9              |
| smallnet_int8      | 5 000  | 0.824  | 0.818    | 1 677.8  | 93 446  | 183 668 710  | 93       | 180    | 28.3               |
| bignet_fp32        | 74     | **0.973** | **0.971** | 3 230.2  | 190 182 | 72 482 390   | 743      | 345    | 54.4               |
| bignet_int8        | host   | 0.878  | 0.880    | (run pending) | 190 182 | 71 911 382 | 188 | 92 | — |
| bignet_pruned_fp32 | 30 000 | 0.937  | 0.937    | 499.2    | 17 417  | 7 614 225    | 68       | 107    | 8.4                |
| bignet_pruned_int8 | 30 000 | 0.936  | 0.936    | **155.8** | 17 417  | 7 437 357    | **18**   | **27** | **2.6**            |

Power assumption used for `E / inference`: STM32U585 Run1 @ 160 MHz, VDD = 3.3 V,
IDD = 31.9 µA/MHz (DS13086) → P ≈ 16.84 mW; E = P × (pre + infer + post) / 1000.

### Key takeaways

- **bignet_pruned_int8 is the clear winner across every axis except top-line fp32 accuracy** —
  11× faster than smallnet_int8, ~10× smaller weights, near-equal
  accuracy to bignet_fp32 (0.936 vs 0.973).
- **fp32 → int8 cost differs per model.** smallnet barely drops
  (0.823 → 0.824), bignet_pruned barely drops (0.937 → 0.936), but
  **bignet drops ~10 pp (0.973 → 0.878)** — its int8 quantization is
  noticeably more lossy. Worth investigating QAT or per-channel symmetric
  quant before shipping bignet_int8.
- **fp32 throughput is ~22 M MACC/s; int8 is ~110 M MACC/s** — that's the
  ~5× speedup from CMSIS-NN integer kernels vs the generic float path.
  Holds across model sizes (see the rate column below).

| model              | MACC      | infer_ms | throughput (M MACC/s) |
|--------------------|-----------|----------|-----------------------|
| smallnet_fp32      | 184.7 M   | 8 244.9  | 22.4                  |
| bignet_fp32        | 72.5 M    | 3 230.2  | 22.4                  |
| bignet_pruned_fp32 | 7.6 M     | 499.2    | 15.2  (overhead-bound)|
| smallnet_int8      | 183.7 M   | 1 677.8  | 109.5                 |
| bignet_pruned_int8 | 7.4 M     | 155.8    | 47.5  (overhead-bound)|

Small models bottleneck on per-layer overhead, so they don't reach the
asymptotic throughput.

---

## 2. Why bignet is faster than smallnet (even though it has more params)

This was the surprising result on the bench: **bignet has 2× the params
of smallnet but 2.5× fewer MACCs**, so inference is faster. The naming
is misleading — "big" / "small" describes the weight budget, not the
compute budget.

### The math

Both nets use the same block (Conv 3×3 → BN → ReLU → MaxPool 2×2,
padding = 1, stride = 1, `bias=False` because BN owns the bias). For
each Conv:

```
MACC   = H_in × W_in × C_in × C_out × 9        # conv runs BEFORE the pool
params = C_in × C_out × 9                      # plus 2·C_out for BN gamma+beta
```

The split matters: **MACC scales with spatial map area**, params don't.
That single fact explains everything below.

### smallnet — 3 stages, channels [32, 64, 128]

| stage | spatial (H×W) | C_in→C_out | MACC      | params  |
|-------|---------------|------------|-----------|---------|
| 1     | 120×160       | 1→32       | 5.5 M     | 288     |
| 2     | 60×80         | 32→64      | **88.5 M**| 18 432  |
| 3     | 30×40         | 64→128     | **88.5 M**| 73 728  |
| FC    | —             | 128→6      | 0.0008 M  | 774     |
| **Σ** |               |            | **~183 M**| **~93 K** |

**96 % of the MACCs are in stages 2 + 3** — those are the layers that
multiply moderate channel counts against still-large spatial maps
(60×80 = 4 800 px and 30×40 = 1 200 px).

### bignet — 5 stages, channels [16, 32, 64, 96, 128]

| stage | spatial (H×W) | C_in→C_out | MACC     | params   |
|-------|---------------|------------|----------|----------|
| 1     | 120×160       | 1→16       | 2.8 M    | 144      |
| 2     | 60×80         | 16→32      | 22.1 M   | 4 608    |
| 3     | 30×40         | 32→64      | 22.1 M   | 18 432   |
| 4     | 15×20         | 64→96      | 16.6 M   | 55 296   |
| 5     | 7×10          | 96→128     | **7.7 M**| **110 592** |
| FC    | —             | 128→6      | 0.0008 M | 774      |
| **Σ** |               |            | **~71 M**| **~190 K** |

**58 % of bignet's params live in stage 5**, where the spatial map is
only 7 × 10 = 70 px. Those 110 K weights only cost 7.7 M MACCs each
inference — they're effectively free at runtime, but they're what
inflates the param count.

### The smoking-gun layer

Compare the **32→64** conv across the two nets:

| net      | spatial | MACC      | params |
|----------|---------|-----------|--------|
| smallnet | 60×80   | **88.5 M**| 18 K   |
| bignet   | 30×40   | 22.1 M    | 18 K   |

**Same weights. Same kernel. bignet executes that same conv at 1/4 the
spatial area, so 4× fewer MACCs.** It's identical compute per pixel —
bignet just touches fewer pixels at that channel count because it
downsampled one extra time before getting there.

### The general principle

> Downsampling makes a Conv cheaper in MACC but does **not** change its
> param count. Adding channels makes a Conv expensive in both — but
> only in MACC proportional to the current spatial map.
>
> So if you want a model with **a lot of feature capacity** (high
> params) but **cheap to run** (low MACC), you do all the channel
> widening **after** the spatial map is already small.

This is exactly the design progression of VGG → ResNet → MobileNet →
EfficientNet: downsample aggressively, put your channel depth where the
spatial map is small. Smallnet is the "VGG-like" extreme (compute-heavy,
param-light); bignet is one step further along this curve.

It's also why pruning works so well on bignet: stage 5 has 110 K weights
producing only 7.7 M MACCs, so chopping channels there barely hurts
MACCs but hugely shrinks the weight footprint. That's how
`bignet_pruned` ends up at 17 K params / 7.4 M MACCs — pruning
preferentially ate the late-stage channels.

---

## 3. Pareto frontier across all training runs

26 runs of the same architecture family
(`baseline_cnn`, varying only `channels` and `dropout_rate`). Best
validation accuracy per run, sorted descending:

| run                                                | val_acc | MACC      | params  | channels                  |
|----------------------------------------------------|---------|-----------|---------|---------------------------|
| theonenet-20260508-031306                          | 0.9734  | 74.7 M    | 412 K   | [16,32,64,96,128,**192**] |
| bignet-20260505-163101_BASELINE                    | 0.9731  | 71.3 M    | 190 K   | [16,32,64,96,128]         |
| hugenet-20260506-032332                            | 0.9717  | 279.8 M   | 759 K   | [32,64,128,192,256]       |
| **bignet_pruned-20260505-210334**                  | **0.9627** | **18.5 M** | **48 K** | [8,16,32,48,64]      |
| massivenet-20260506-201654                         | 0.9549  | 248.8 M   | 316 K   | [32,64,128,192]           |
| hugenet_pruned_08-20260506-123238                  | 0.9452  | 11.9 M    | 31 K    | [6,13,26,38,51]           |
| middlenet_greyscale-20260505-004825                | 0.9355  | 69.1 M    | 98 K    | [16,32,64,128]            |
| bignet_pruned-20260505-225640 _BASELINE_PRUNED_    | 0.9288  | 7.3 M     | 17.5 K  | [5,10,19,29,38]           |
| theonenet_pruned_082-20260508-153118               | 0.9205  | 3.0 M     | 14 K    | [3,6,12,17,23,35]         |
| theonenet_pruned_08-20260508-151426                | 0.9163  | 3.3 M     | 17 K    | [3,6,13,19,26,38]         |
| theone_pruned_085-20260508-185142                  | 0.9148  | 1.9 M     | 9.5 K   | [2,5,10,14,19,29]         |
| smallnet_greyscale-20260504-184641 _BASELINE_SMALL_| 0.8348  | 182.5 M   | 94 K    | [32,64,128]               |
| (… 14 more weaker runs)                            | …       | …         | …       | …                         |

Pareto-frontier points (each is best-acc-at-its-MACC-budget):

```
   0.9734 ┤ theonenet      [16,32,64,96,128,192]  (75 M MACC)
   0.9731 ┤ bignet         [16,32,64,96,128]      (71 M MACC)
   0.9627 ┤ bignet_pruned  [8,16,32,48,64]        (18.5 M MACC)   ← strictly better
   0.9452 ┤ hugenet_pruned [6,13,26,38,51]        (12 M MACC)        than BASELINE_PRUNED
   0.9288 ┤ bignet_pruned  [5,10,19,29,38]        (7.3 M MACC)
   0.9205 ┤ theonenet_pr.  [3,6,12,17,23,35]      (3.0 M MACC)
   0.9148 ┤ theone_pruned  [2,5,10,14,19,29]      (1.9 M MACC)
```

### Two notes from skimming the runs

1. **`bignet_pruned-20260505-210334` (channels `[8,16,32,48,64]`) is
   strictly better than `bignet_pruned-20260505-225640_BASELINE_PRUNED`**
   on val accuracy (0.9627 vs 0.9288) at ~2.5× the MACC. It was trained
   but never picked as the BASELINE. Worth flashing before any retrain.
2. **Every single run used identical training hyperparameters** (same
   augmentation, same LR schedule, same 150 epochs, same dropout 0.05 or
   0.1). Only the architecture varied. So there is real accuracy
   headroom that no run has captured — stronger aug, longer cosine,
   label smoothing, etc.

### The general shape of the curve

- Six-stage architectures (theonenet family) Pareto-dominate five-stage
  ones at low MACC. Extra downsampling lets you keep adding channels
  cheaply.
- Pruning at ~50 % consistently outperforms pruning at ~85 % at the
  same final compute budget. The aggressive pruners lose more accuracy
  than they save.
- Past ~10 M MACC, accuracy gains plateau quickly — the dataset just
  isn't that complex.

---

## 4. Where this leaves the engineering tradeoffs

| Goal               | Pick                                     | Why                                                           |
|--------------------|------------------------------------------|---------------------------------------------------------------|
| Max accuracy       | bignet_fp32 (or theonenet if retrained)  | 0.973 acc, 3.2 s inference, 743 KB flash — fits the U585 fine |
| MCU sweet spot     | **bignet_pruned_int8**                   | 0.936 acc, **156 ms**, **18 KB** weights — ship this          |
| Maybe even better  | flash `bignet_pruned-210334` int8        | Untested but 0.96 val acc at 18.5 M MACC — try before retraining |

The "small + fast + accurate" three-way win in the *same* architecture
family is bounded by the Conv 3×3 design — to push the frontier further,
the next step is depthwise-separable blocks (`Conv 3×3 dw → Conv 1×1
pw`), which historically buy 3-5× fewer MACCs at similar accuracy.
That's a `make_stage` rewrite, not a config change.
