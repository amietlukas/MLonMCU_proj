# Ball Detection (STYOLO-Style, ONNX-Friendly)

This module implements a complete training/eval/export pipeline for single-class ball detection using mixed datasets (`SPLDataset` + `OurDataset`).

## Implemented in this version

- Multi-source dataset parser for:
  - `datasets/BALL/SPLDataset`
  - `datasets/BALL/OurDataset`
- Reproducible train/val split generation + reuse, stratified per source
- Image-driven dataset indexing (all image files are included, not only annotated rows)
- Empty-image support (images without annotation rows are valid no-ball samples)
- Mixed-source train batching (every train batch contains samples from both datasets)
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
  - center error (mean / median / p95) in pixels
  - normalized center error by image diagonal (mean / median / p95)
  - false positives per image
  - small/medium/large recall bins
- ONNX export (`fp32.onnx`) with output heads `p8`, `p16`, `p32`
- INT8 PTQ QDQ export (`int8_ptq_qdq.onnx`) via ONNX Runtime static quantization

## Dataset layout expected

Dataset root config points to `datasets/BALL`.
By default, the config uses two dataset sources:

- `datasets/BALL/SPLDataset/<scene>/annotations.csv`
- `datasets/BALL/OurDataset/<scene>/annotations.csv`

Rows are parsed as:

- `filename,x,y,width,height`

And interpreted as:

- `x, y` = top-left corner in pixels
- `width, height` = box size in pixels

If the same image appears multiple times in CSV rows, rows are merged (`duplicate_policy: append`) so one image can have multiple ball boxes.
Images that exist in a source folder but have no annotation row are kept as valid empty (no-ball) training/eval samples.
This is important for learning the negative/no-ball case.

Backward compatibility is kept: if `dataset.sources` is omitted, the loader falls back to legacy SPL multiclass CSV parsing (`image,class,x,y,w,h,...`) via `paths.images_dir`.

## Split files

Split files are written to:

- `software/ball_detection/splits/train.txt`
- `software/ball_detection/splits/val.txt`

If `dataset.reuse_splits: true` and both files exist, they are reused.
Split validity is checked against the current dataset IDs and source coverage.
If invalid (for example after switching datasets), splits are regenerated automatically.
Splits are always generated over all images, not only annotated rows.

### Batch source mixing

With `dataset.mix_sources_per_batch: true`, train batches are source-balanced.
For two sources and batch size `32`, each batch contains `16` samples from each source.
This guarantees per-batch source presence, but can oversample the smaller source within an epoch.

## Main config

- `software/ball_detection/configs/ball_styolo_nano.yaml` (multi-ball profile, `max_detections=3`)
- `software/ball_detection/configs/ball_styolo_nano_oneball.yaml` (single-ball profile, top-1 without NMS)

Key settings:

- input (default): `640x480`, `rgb`, `3` channels, `resize`
- alternative preprocessing: `letterbox` (if aspect-ratio-preserving square input is desired)
- data sources: `dataset.sources` list (per-source path + annotation format + bbox format)
- split reuse control:
  - config: `dataset.reuse_splits`
  - CLI override: `--reuse-splits` / `--regenerate-splits`
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

Force a fresh split generation:

```bash
python -m ball_detection.train \
  --config ball_detection/configs/ball_styolo_nano.yaml \
  --name ball_regen_split \
  --regenerate-splits
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

Optional split control is also available for eval:

- `--reuse-splits`
- `--regenerate-splits`

## Export only

```bash
python -m ball_detection.export \
  --config ball_detection/configs/ball_styolo_nano.yaml \
  --checkpoint ball_detection/runs/<run-id>/checkpoints/best.pt \
  --device auto
```

Optional split control is also available for export:

- `--reuse-splits`
- `--regenerate-splits`

## Prediction bbox format

- Internal training/assignment/IoU logic uses `xyxy` boxes.
- Final postprocessed predictions now expose both:
  - `boxes` in `xyxy`
  - `boxes_xywh_topleft` in top-left + width/height format
- `boxes_xywh_topleft` is available on `ImagePrediction` objects in `src/training/metrics.py`.
- Dataset targets now also carry both forms:
  - `target["boxes"]` (`xyxy`)
  - `target["boxes_xywh_topleft"]`
  - `target["boxes_orig"]` (`xyxy` in original image)
  - `target["boxes_orig_xywh_topleft"]`

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
