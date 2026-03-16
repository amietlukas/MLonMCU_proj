# MLonMCU Software

This folder contains the ML pipelines used in this project, split into separate model tracks with shared utilities.

## Structure

```text
software/
├── classifier/         # Gesture classification pipeline
├── keypoints/          # Hand keypoint extraction pipeline
├── utils/              # Shared modules (config/checkpoint/run)
├── tools/              # Project tools and notebooks
├── requirements.txt
└── webcam_classify.py
```

## Model Tracks

- Classifier details: see classifier/README.md
- Keypoints details: see keypoints/README.md

## Setup

From repository root:

```bash
cd software
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Commands

Train classifier:

```bash
python classifier/main.py --name baseline128
```

## Notes

- Keep model-specific code inside each model folder.
- Put reusable code only in utils/.
- Keep output folders model-local (classifier/runs/, later keypoints/runs/).
