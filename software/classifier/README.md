# Classifier Pipeline

This folder contains the hand gesture classification model and training pipeline.

## Folder Layout

```text
classifier/
├── config.yaml         # Classifier config
├── main.py             # Training orchestration entrypoint
├── src/                # Classifier-specific modules
└── runs/               # Training outputs
```

## What This Model Does

Classifies HAGRID hand gestures into 6 classes:

- palm
- rock
- pinkie
- one
- fist
- others

## Train

From software/:

```bash
python classifier/main.py --name baseline128
```

Resume from checkpoint:

```bash
python classifier/main.py --name baseline128_continued --checkpoint baseline128-YYYYMMDD-HHMMSS
```

## Config

Main config file:

- classifier/config.yaml

Important:

- data.dataset_root in this file is relative to classifier/.

## Outputs

Each run creates:

```text
classifier/runs/<run_name>-<timestamp>/
├── config_snapshot.yaml
├── model_summary.txt
├── code_snapshot.zip
├── checkpoints/
│   └── best.pt
└── logs/
    ├── metrics.csv
    ├── terminal.log
    └── (plots/csv diagnostics)
```

## Shared vs Local Code

Shared modules imported from utils/:

- utils/config.py
- utils/checkpoint.py
- utils/run.py

Classifier-specific modules remain in classifier/src/.
