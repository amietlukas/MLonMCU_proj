# Ball Detection (STYOLO-Style, ONNX-Friendly)

This module implements a first complete training/eval/export pipeline for single-class ball detection on SPL-style CSV annotations.

## Implemented in this version

- Dataset parser for `datasets/BALL/SPLBallDataset/full_size_images`
- Single-class filtering (`Ball`) with robust CSV parsing and invalid-box warnings
- Reproducible train/val split generation + reuse
- Configurable resize policy (`letterbox` or direct `resize`) with consistent box transform metadata
- Optional train augmentations (hflip, color jitter, blur)
- Optional utility to export parsed labels in YOLO txt format (`export_yolo_labels`)
- STYOLO-style compact detector (strides `8/16/32`)
- Anchor-free grid assignment and BCE+IoU training loss
- Validation metrics:
  - `mAP@0.5`
  - `mAP@0.5:0.95`
  - precision / recall / F1
  - mean IoU (matched TPs)
  - center error in pixels
  - false positives per image
  - small/medium/large recall bins
- ONNX export (`fp32.onnx`) with output heads `p8`, `p16`, `p32`
- INT8 PTQ QDQ export (`int8_ptq_qdq.onnx`) via ONNX Runtime static quantization

## Dataset layout expected

Dataset root config points to `datasets/BALL`.
Images and CSVs are read from:

- `datasets/BALL/SPLBallDataset/full_size_images/<scene_folder>/<scene_name>.csv`
- `datasets/BALL/SPLBallDataset/full_size_images/<scene_folder>/<image files>`

Each CSV row is parsed as:

- `image_name, class_name, x, y, w, h, class_name, x, y, w, h, ...`

In current config (`dataset.bbox_format: cxcywh_radius`) these are interpreted as:

- `x, y` = box center in pixels
- `w, h` = radius-like half-size in pixels (dataset-specific behavior)

Only `class_name == Ball` is kept.
Other classes are ignored.
Rows with no `Ball` are supported as zero-object images.
If the same image appears multiple times in CSV rows, the default is `dataset.duplicate_policy: last`.

## Split files

Split files are written to:

- `software/ball_detection/splits/train.txt`
- `software/ball_detection/splits/val.txt`

If `dataset.reuse_splits: true` and both files exist, they are reused.

## Main config

- `software/ball_detection/configs/ball_styolo_nano.yaml` (multi-ball profile, `max_detections=3`)
- `software/ball_detection/configs/ball_styolo_nano_oneball.yaml` (single-ball profile, top-1 without NMS)

Key settings:

- input (default): `640x480`, `rgb`, `3` channels, `resize`
- alternative preprocessing: `letterbox` (if aspect-ratio-preserving square input is desired)
- model: `ball_styolo_nano`, strides `[8,16,32]`
  - compactness/accuracy knobs: `model.width_mult`, `model.neck_out_ch`
- loss/assignment knobs for convergence:
  - `loss.obj_loss_mode`, `loss.obj_pos_weight`, `loss.obj_neg_weight`, `loss.obj_bce_pos_weight`
  - `loss.assign_center_radius` (0=center cell only, 1=3x3 neighborhood)
- export: ONNX opset 13, optional INT8 QDQ

## Training

Run from `software/` directory.

```bash
python -m ball_detection.train --config ball_detection/configs/ball_styolo_nano.yaml --name ball_nano --device auto
```

Equivalent script form:

```bash
python ball_detection/train.py --config ball_detection/configs/ball_styolo_nano.yaml --name ball_nano --device auto
```

Debug quick pass:

```bash
python -m ball_detection.train \
  --config ball_detection/configs/ball_styolo_nano.yaml \
  --name ball_debug_cpu \
  --device cpu \
  --max-train-batches 2 \
  --max-val-batches 2
```

### Training artifacts per run

Written to:

- `software/ball_detection/runs/YYYYMMDD-HHMMSS-<name>/`

Includes:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `training.log`
- `metrics.csv`
- `metrics.svg` (training dashboard generated from `metrics.csv`)
- `config_snapshot.yaml`
- `model_summary.txt`
- `groundtruth_preview/*.png`
- `predictions_preview/epoch_*/...png`
- `exports/fp32.onnx`
- `exports/int8_ptq_qdq.onnx` (if quantization succeeds)
- `exports/calibration_report.json` (if quantization succeeds)

## Preview folders explained

- `groundtruth_preview/`
  - Saved once at startup from train and val batches (`train_gt_*`, `val_gt_*`).
  - Only ground-truth boxes are drawn (`lime`).
- `predictions_preview/epoch_XXX/`
  - Saved during validation (first batch per epoch).
  - Ground truth is `lime`, predictions after confidence+NMS are `red`.

If older custom debug folders exist (for example under an old `runs/smoke_checks` run), treat them as historical debugging artifacts, not as the current pipeline reference.

## Evaluation

```bash
python -m ball_detection.evaluate \
  --config ball_detection/configs/ball_styolo_nano.yaml \
  --checkpoint ball_detection/runs/<run-id>/checkpoints/best.pt \
  --device auto
```

## Export only

```bash
python -m ball_detection.export \
  --config ball_detection/configs/ball_styolo_nano.yaml \
  --checkpoint ball_detection/runs/<run-id>/checkpoints/best.pt \
  --device auto
```

## ONNX / deployment notes

- ONNX graph intentionally contains raw head outputs only (`p8/p16/p32`).
- Postprocessing stays outside graph (decode, threshold, NMS, scaling).
- INT8 uses ONNX Runtime static QDQ quantization.
- If `onnx` or `onnxruntime` is missing, export/quantization is skipped with `[WARN]` instead of crashing.

## Known limitations

- First version assignment is simple center-cell matching (not SimOTA/QAT).
- Only `ball_styolo_nano` is implemented right now.
- TinyissimoYOLO and additional STYOLO variants are not implemented yet.
- No ONNX-side NMS or decode by design.
