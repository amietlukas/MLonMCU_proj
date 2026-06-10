# MLonMCU Software

This folder contains the ML pipelines used in this project, split into separate model tracks with shared utilities.

## Structure

```text
software/
├── ball_detection/     # Ball detection pipeline
├── classifier/         # Hand Gesture classification pipeline
├── utils/              # Shared modules (config/checkpoint/run)
├── tools/              # Dataset notebooks + RGB->grayscale helper
└── requirements.txt
```

## Model Tracks

- Ball Detection details: see [ball_detection/README.md](ball_detection/README.md)
- Classifier details: see [classifier/README.md](classifier/README.md)

## Setup

From repository root:

```bash
cd software
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Commands

Train the classifier:

```bash
python classifier/main.py --name baseline128
```

Train the ball detector:

```bash
python -m ball_detection.train --config ball_detection/configs/ball_styolo_nano.yaml --name ball_nano --device auto
```

## Notes

- Keep model-specific code inside each model folder.
- Put reusable code only in utils/.
- Keep output folders model-local (`classifier/runs/`, `ball_detection/runs/`).
