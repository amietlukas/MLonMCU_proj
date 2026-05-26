# PowerPoint Prompt — MLonMCU Final Presentation

> Paste the block below into the Claude / Copilot integration inside PowerPoint to generate the deck. The figures referenced live in this repo at the paths given — drop them onto the corresponding slides after generation.

---

## PROMPT — copy from here

You are helping me build an academic project presentation for **ETH Zürich** titled:

**"MLonMCU — Embedded Vision for Gesture-Controlled RC-Car Steering & Active Ball Tracking"**

Audience: ML / embedded-systems faculty and peers. Tone: technical, concise, engineering-focused. No marketing fluff. Use compact bullets (≤ 12 words per bullet, ≤ 5 bullets per slide). Numbers, units, and model names must be exact — do not round or invent. Use a clean serif/sans-serif ETH-style theme (dark blue accents, white background, monospaced font for numbers/code). Each slide has a short title (≤ 8 words) and an optional one-line subtitle.

Generate **exactly 18 slides** in the order below. For every slide, output: (a) Title, (b) Subtitle (optional), (c) Bullets, (d) Speaker notes (~3–5 sentences), (e) Suggested figure path from the list at the bottom — DO NOT invent figure paths.

---

### PROJECT FACT SHEET (use these verbatim — do not paraphrase the numbers)

**Hardware**
- Gesture MCU: **B-U585I-IOT02A** — STM32U585 (Cortex-M33, 160 MHz, 768 KB Flash, 256 KB SRAM)
- Camera: **B-CAMS-OMV** with **OV5640** sensor, QQVGA YUV422 (160×120), Y-channel direct to grayscale
- Drive MCU: **STM32 N6** mounted on the RC car ("Paul")
- Link: UART/serial-over-Bluetooth bridge, single-byte command protocol (`0` stop, `1` fwd, `2` fwd-right, `3` fwd-left, `4` reverse, `5` hold-last)
- X-CUBE-AI **10.2.0** for U5 deployment; ONNX QDQ INT8 path

**Gesture classifier — HAGRID dataset**
- 6 classes: `palm` (stop), `rock` (drive-fwd-straight), `pinkie` (steer-fwd-right), `one` (steer-fwd-left), `fist` (drive-bwd-straight), `others` (background)
- 141,258 train / 18,000 val / 30,000 test, 8-bit grayscale, 160×120 QQVGA
- Per-channel norm folded into Conv1 → MCU receives raw uint8

**Gesture classifier — architecture family (VGG-inspired)**
- Block: `Conv3×3 → BN → ReLU → MaxPool2×2`, head = Global Average Pool → Linear → 6-way softmax
- **SmallNet**: 3 stages, channels `[32, 64, 128]`, **93,446 params, 184.7 M MACCs**
- **BigNet**: 5 stages, channels `[16, 32, 64, 96, 128]`, **190,182 params, 72.5 M MACCs**
- **BigNet-pruned**: structured L1-norm channel prune @50%/stage → channels `[5, 10, 19, 29, 38]`, **17,417 params, 7.4 M MACCs**
- **Surprising result**: BigNet has 2× SmallNet's params but **2.5× fewer MACCs** AND higher accuracy — params live in late stages (small spatial maps), compute in early stages

**Gesture classifier — training**
- Adam, lr 3e-3, wd 1e-5, CosineAnnealing T_max 150, eta_min 1e-5, CE loss, batch 128, 150 epochs, early-stop patience 20
- PTQ via ONNX Runtime static quantizer (per-tensor symmetric, min-max calibration, 32 batches ≈ 4096 images)
- Pruning is **2-stage**: (1) L1 channel-prune, accuracy drops sharply, (2) fine-tune 150 epochs → recovers near-original accuracy

**Gesture classifier — measured results (STM32U585 @ 160 MHz)**
| Model | Acc FP32 | Acc INT8 | Latency INT8 | Flash INT8 | RAM INT8 | Energy INT8 |
|---|---|---|---|---|---|---|
| SmallNet | 0.823 | 0.824 | 1677.8 ms | 93 KB | 180 KB | 28.3 mJ |
| BigNet | 0.977 | 0.878 | ~723 ms (host) | 188 KB | 92 KB | — |
| **BigNet-pruned** | **0.937** | **0.936** | **155.8 ms** | **17.6 KB** | **27 KB** | **2.6 mJ** |
- BigNet-pruned-INT8 is the deployed model: **11× faster, 10× smaller than SmallNet-INT8**, with **+11 pp accuracy**

**Ball detector — dataset**
- Hybrid: self-collected **OurDataset** + **RoboCup SPLDataset**
- 2,697 train / 676 val, **640×480 RGB**, single class "ball", hard-negative no-ball frames retained
- CSV labels (`filename, x, y, w, h`, top-left + size)
- Stratified per source; mixed-source batching (16 SPL + 16 ours per batch of 32)
- Augmentations: H-flip 0.5, color jitter ±0.2, Gaussian blur p=0.15

**Ball detector — architecture (STYolo-Nano, YoloX-family, anchor-free)**
- **Backbone** `STYOLONanoBackbone`: stem (s=2) + 4 residual-bottleneck stages, outputs at **strides 8 / 16 / 32**
- **Neck** `STYOLONeck`: learnable feature fusion, projects to uniform **96-ch** intermediate
- **Head** `STYOLOHead`: three parallel 1×1 conv decoders → **5 channels per cell** `[tx, ty, tw, th, obj]` — no class logits (single class)
- Single-class optimization removes class-branch FLOPs and CE loss entirely

**Ball detector — label assignment (two strategies compared)**
- **Center Assigner** (baseline): GT center cell + `center_radius=1` neighborhood (3×3) → positive, area-tiebreak
- **SimOTA-Lite** (experimental, deployed): per-GT candidate set across all strides; cost = `3.0·(1-IoU) + 1.0·BCE_obj→1 + 0.5·norm_dist`; top-k dynamic selection; lower-cost GT wins ties
- Box loss: **DIoU** beat plain IoU empirically (center-distance term ⇒ tighter localization)
- Obj loss: focal BCE (α 0.25, γ 2.0)
- `loss = 1.0·L_obj + 5.0·L_box`

**Ball detector — training**
- AdamW lr 1e-3, wd 1e-4, Cosine T_max 150 eta_min 1e-6, batch 32, early-stop patience 30 on mAP@0.5:0.95

**Ball detector — best run (`simota_diou_v2`, epoch 150)**
- **mAP@0.5 = 93.7 %**, mAP@0.5:0.95 = 73.7 %, precision 95.86 %, recall 86.29 %, F1 = 90.82 %
- Mean IoU 0.894, mean center-error **2.01 px** (median 1.52, p95 5.10)
- False positives per image: **0.037**
- ONNX size: **8.0 MB FP32 → 2.2 MB INT8** (PTQ QDQ, –73 %)
- Latest pruned variant: `simota_diou_v2_pruned_30` (30 % structured prune)

**System integration**
- U5 runs gesture classifier on captured frame → maps argmax to drive command → sends single ASCII byte over BLE/UART
- N6 receives byte → sets PWM/direction per wheel → drives motors (stateless, no closed loop on the wire)
- Active ball tracking: N6-side detection planned to override gesture command when ball is locked (centroid-error → steering bias)

---

### SLIDE-BY-SLIDE CONTENT

**Slide 1 — Title**
- Title: "MLonMCU — Embedded Vision for Gesture-Controlled RC-Car Steering & Active Ball Tracking"
- Subtitle: ETH Zürich · Spring 2026
- Bullets: authors, course, date (leave placeholders).
- Figure: none (clean title) or a hero photo of the RC car if available.
(dont change current title)

**Slide 2 — Motivation**
- Why ML on MCUs matters: edge inference, sub-watt power budget, low latency, privacy, no cloud. -> very short, if even mention! mainly focus on motivation of this exact project. why is hand gestured RCcar steering desired? why together with ball detection? be creative
- Trade-off triangle: **accuracy ↔ latency ↔ energy** under hard memory limits.
- Robotics needs *both* perception and control in real time on the same device class.
- Speaker notes: explain why a microcontroller (not a Jetson/Pi) — it forces honest engineering on the model side and is representative of where real embedded products live.
- mainly bring motivation for the project.

**Slide 3 — Project Goal**
- Two coordinated MCUs on one robot platform.
- **U5 + B-CAMS-OMV** → gesture-driven teleoperation of an RC car.
- **N6 + camera** → onboard ball detection & active ball-lock while driving.
- Cross-MCU command bridge over BLE/UART.
- Deliver models that satisfy **acc / latency / energy** constraints simultaneously.
- be as small and fast as possible
- Figure: system block diagram (gesture-MCU ⇒ BLE ⇒ drive-MCU + camera ⇒ motors). If none exists, describe it textually for the slide author to draw.

**Slide 4 — Hand Gesture Recognition — overview**
- 6 classes mapped to drive commands (palm/rock/pinkie/one/fist/others).
- Greyscale 160×120 input from OV5640 Y-plane (no resize).
- VGG-style CNN (Conv–BN–ReLU–MaxPool blocks + GAP head).
- Pipeline: collect → train FP32 → prune → fine-tune → PTQ INT8 → deploy via X-CUBE-AI 10.2.0.

**Slide 5 — Hand Gesture: Dataset (HAGRID)**
- Source: HAGRID, resized to QQVGA grayscale (`hagrid_full_qqvga_resize`).
- Splits: **141 258 train / 18 000 val / 30 000 test** (balanced per class).
- 6 classes selected to map cleanly to drive intents.
- Per-channel normalisation (μ=0.5427, σ=0.2425) **folded into Conv1** → MCU feeds raw uint8.
- Figure: a 2×3 sample-grid of one image per class would be ideal — note for slide author to drop in.

**Slide 6 — Hand Gesture: Model / Implementation**
- SmallNet — 3 stages `[32,64,128]`, 93 K params, **185 M MACCs** — params-light, compute-heavy (early big maps).
- BigNet — 5 stages `[16,32,64,96,128]`, 190 K params, **72 M MACCs** — params concentrated in late small maps.
- Counter-intuitive: BigNet has **more params but fewer MACCs** AND higher accuracy.
- Conclusion: depth + downsampling > width on early maps for our budget.
- Figure: `presentation/classifier/plots/model_architectures/smallnet_greyscale_baseline_small_architecture.svg` + `bignet_baseline_architecture.svg` side-by-side.

**Slide 7 — Hand Gesture: Model / Implementation (cont.)**
- **Pre-processing fusion**: `W' = W/σ`, bias absorbs `255·μ` → zero runtime norm cost.
- **Structured L1-norm channel pruning** @50 %/stage on trained BigNet → channels `[5,10,19,29,38]`.
- **Two-stage**: prune (acc collapses) → fine-tune 150 epochs (acc recovers to 0.937).
- **PTQ INT8** via ONNX-Runtime QDQ, per-tensor symmetric, min-max calibration; no QAT needed (negligible drop on pruned model).
- Deployed model fits in **17.6 KB Flash / 27 KB RAM**.
- Figure: `presentation/classifier/plots/model_architectures/bignet_pruned_baseline_pruned_architecture.svg`.

**Slide 8 — Hand Gesture: Results**
- BigNet-pruned-INT8 wins: 0.936 acc, **156 ms**, **2.6 mJ**, **17.6 KB**.
- vs SmallNet-INT8 (0.824, 1678 ms, 93 KB): +11 pp acc, 11× faster, 5× smaller, 11× less energy.
- vs BigNet-INT8 (0.878): pruning + retrain recovered the INT8 gap that hit dense BigNet.
- Figures (pick two): `presentation/classifier/plots/model_comparison/accuracy_vs_inference_latency.svg`, `accuracy_vs_flash_footprint.svg`.

**Slide 9 — Ball Detection — overview**
- Anchor-free single-class detector → "Paul" the RC car tracks a ball while driving.
- Architecture: **STYolo-Nano** (YoloX-family, anchor-free, FCOS-style).
- Trained on **OurDataset** + **RoboCup SPL** images.
- Two assigners + two box-loss variants compared; SimOTA + DIoU wins.

**Slide 10 — Ball Detection: Dataset**
- 2 697 train / 676 val frames, 640×480 RGB, single class "ball".
- Hybrid sources: self-collected (Roboflow-labelled, converted CSV) + SPL benchmark.
- Hard-negative frames (no ball) retained for robustness.
- Mixed-source batching (16+16) avoids SPL dominance.
- Augmentations: H-flip 0.5, color-jitter ±0.2, blur p=0.15.

**Slide 11 — Ball Detection: Model / Implementation**
- **Backbone**: residual-bottleneck stages, outputs at **strides 8/16/32**.
- **Neck**: learnable fusion → 96-ch uniform features.
- **Head**: 5 channels per cell `[tx, ty, tw, th, obj]` — no class logits (single class).
- **Assigners**: Center (baseline 3×3) vs **SimOTA-Lite** (cost = 3·(1-IoU) + 1·BCEobj + 0.5·dist).
- **Box loss**: **DIoU > IoU** (center penalty ⇒ tighter localisation).
- Obj loss: focal BCE (α 0.25, γ 2.0); total = 1·L_obj + 5·L_box.
- PTQ INT8: **8.0 MB → 2.2 MB** ONNX.

**Slide 12 — Ball Detection: Results**
- Best: `simota_diou_v2` — **mAP@0.5 = 93.7 %**, mAP@0.5:0.95 = 73.7 %, P 95.9 %, R 86.3 %.
- Mean centre-error **2.0 px** at 640×480 → sub-degree steering correction is feasible.
- FP/image ≈ 0.04 — low false-trigger rate critical for closed-loop control.
- Latest 30 %-pruned variant retains accuracy at lower compute.
- Figure: `software/ball_detection/runs/20260521-195948-simota_diou_v2_pruned_30/metrics.svg` (training curves) + a sample from `predictions_preview/`.

**Slide 13 — Full System**
- U5 captures frame → INT8 classifier → argmax → command byte over BLE/UART → N6 sets PWM/direction.
- N6 captures own frame → INT8 detector → centroid → steering correction overlaid on gesture command.
- Two-MCU split: perception/control decoupling, each MCU sized to its workload.
- Power-budgeted entirely from the RC battery.
- Figure: system architecture diagram (block diagram with both MCUs, cameras, BLE link, motors).

**Slide 14 — Project Goals & Future Work**
- Achieved: trained + quantized + pruned both models; on-MCU gesture inference verified.
- Open: full N6 firmware + camera bring-up; on-MCU detector benchmarks; closed-loop ball-lock controller.
- Stretch: CMSIS-NN/TFLite-Micro path, 1.8 V low-power BigNet run, demo video.
- Bullet from TODO: "N6 kaufen, Präsentation, Kommunikation zwischen STM32s, Demo-Video".

**Slide 15 — Demo**
- Placeholder slide for live or recorded demo (gesture → drive, ball-lock).
- Speaker notes: cue the recording or describe the live demo flow (palm = stop, rock = forward, etc.).

**Slide 16 — Results: Memory Usage**
- Classifier (INT8): SmallNet 93 KB / BigNet 188 KB / **BigNet-pruned 17.6 KB** Flash.
- Detector: 8.0 MB FP32 → **2.2 MB INT8**.
- RAM (classifier INT8): 27–180 KB; pruned fits well within U5's 256 KB SRAM.
- Figure: `presentation/classifier/plots/model_comparison/memory_footprint_dual_axis.svg`.

**Slide 17 — Results: Metrics**
- Classifier: 0.936 acc / 0.936 macro-F1 for BigNet-pruned-INT8 (+11 pp over SmallNet-INT8).
- Detector: mAP@0.5 93.7 %, mAP@0.5:0.95 73.7 %, P 95.9 %, R 86.3 %, mean IoU 0.894.
- Quantisation cost: ≤ 0.001 on pruned classifier, ~0.04 on dense BigNet (why we prune+retrain before PTQ).
- Figure: `presentation/classifier/plots/model_comparison/classification_quality_2x2.svg` and `macro_f1_spotlight.svg`.

**Slide 18 — Results: Latency & Power Consumption**
- Latency (STM32U585 @ 160 MHz, INT8): SmallNet 1678 ms · BigNet ~723 ms · **BigNet-pruned 156 ms**.
- Energy / inference: 28.3 mJ → **2.6 mJ** (11×) for BigNet-pruned-INT8.
- Pre/post overhead negligible (< 1 ms) thanks to norm-fused Conv1.
- Detector latency on N6: pending hardware bring-up.
- Figures: `presentation/classifier/plots/model_comparison/accuracy_vs_inference_latency.svg` + `latency_pre_infer_post.svg`.

---

### FIGURE LIBRARY (use only these — full paths relative to the repo root)

Classifier architecture diagrams:
- `presentation/classifier/plots/model_architectures/smallnet_greyscale_baseline_small_architecture.{svg,pdf,png}`
- `presentation/classifier/plots/model_architectures/bignet_baseline_architecture.{svg,pdf,png}`
- `presentation/classifier/plots/model_architectures/bignet_pruned_baseline_pruned_architecture.{svg,pdf,png}`

Classifier comparison plots:
- `presentation/classifier/plots/model_comparison/accuracy_vs_inference_latency.{svg,pdf,png}`
- `presentation/classifier/plots/model_comparison/accuracy_vs_flash_footprint.{svg,pdf,png}`
- `presentation/classifier/plots/model_comparison/macro_f1_vs_inference_latency.{svg,pdf,png}`
- `presentation/classifier/plots/model_comparison/macro_f1_per_inference_ms.{svg,pdf,png}`
- `presentation/classifier/plots/model_comparison/macro_f1_spotlight.{svg,pdf,png}`
- `presentation/classifier/plots/model_comparison/classification_quality_2x2.{svg,pdf,png}`
- `presentation/classifier/plots/model_comparison/latency_pre_infer_post.{svg,pdf,png}`
- `presentation/classifier/plots/model_comparison/memory_footprint_dual_axis.{svg,pdf,png}`

Ball-detection training curves & predictions:
- `software/ball_detection/runs/20260521-195948-simota_diou_v2_pruned_30/metrics.svg`
- `software/ball_detection/runs/20260520-144405-simota_diou_v2/metrics.svg`
- `software/ball_detection/runs/<run>/predictions_preview/*.jpg` (qualitative samples)
- `software/ball_detection/runs/<run>/groundtruth_preview/*.jpg`

System / hero photos: not yet captured — leave placeholders on slides 1, 3, 13, 15.

---

### STYLE RULES

- One idea per slide; bullets are sentence fragments, not paragraphs.
- Always write numbers with units (`156 ms`, `17.6 KB`, `2.6 mJ`).
- Refer to MCUs as **STM32U5 / STM32N6**, not "U5/N6" alone, on first mention per slide.
- Never refer to QAT — we did **PTQ only**.
- Never claim the N6 firmware is complete — it is in bring-up.
- Speaker notes: 3–5 sentences, plain prose, explain the *why* behind each bullet.
- Reserve one accent colour for "deployed model" highlights (BigNet-pruned-INT8 row in tables, etc.).


- mention if you miss anything in the chat

Produce the deck now.
