# MLonMCU Hand Gesture Classification Project

A PyTorch-based deep learning project for hand gesture classification on the HAGRID dataset, designed for autonomous vehicle control.

## 🎯 Project Overview

This project trains a Convolutional Neural Network (CNN) to classify 6 hand gestures that map to vehicle control commands:

| Gesture | Class ID | Control Command |
|---------|----------|-----------------|
| palm    | 0        | STOP |
| rock    | 1        | DRIVE FORWARD STRAIGHT |
| pinkie  | 2        | STEER FORWARD RIGHT |
| one     | 3        | STEER FORWARD LEFT |
| fist    | 4        | DRIVE BACKWARD STRAIGHT |
| others  | 5        | OTHER/background |

## 📁 Project Structure

changed! write new structure TO DO

<!-- ```
MLonMCU_proj/
├── config.yaml              # All hyperparameters and settings
├── main.py                  # Main training orchestrator
├── requirements.txt         # Python dependencies
├── datasets/                # Dataset directory
│   └── hagrid_balanced_classification/
│       ├── train/          # Training images (by class)
│       ├── val/            # Validation images (by class)
│       └── test/           # Test images (by class)
├── src/                     # Source code modules
│   ├── data.py             # Dataset loading, transforms, dataloaders
│   ├── model.py            # CNN architecture (BaselineCNN)
│   ├── engine.py           # Training & evaluation loops
│   ├── checkpoint.py       # Model checkpoint management
│   ├── metrics.py          # CSV logging for metrics
│   ├── per_class_metrics.py # Confusion matrix & per-class metrics
│   ├── config.py           # Config validation
│   ├── run.py              # Run directory management
│   ├── model_utils.py      # Model summary generation
│   └── viz.py              # Visualization utilities
└── runs/                    # Training outputs
    └── <run_name>-<timestamp>/
        ├── config_snapshot.yaml
        ├── model_summary.txt
        ├── checkpoints/
        │   └── best.pt
        └── logs/
            ├── metrics.csv
            ├── confusion_matrix.csv
            ├── per_class_metrics.csv
            └── transform_preview_train.png
``` -->

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository_url>
cd MLonMCU_proj

# Install dependencies
pip install -r requirements.txt
```

### Training

**Start new training:**
```bash
python main.py --name baseline128
```

**Resume from checkpoint:**
```bash
python main.py --name baseline128_continued --checkpoint baseline128-20260222-123456
```

### Configuration

Edit `config.yaml` to customize:
- Image size: `data.input_size` (default: 128)
- Batch size: `train.batch_size` (default: 512)
- Learning rate: `train.lr` (default: 3e-3)
- Epochs: `train.epochs` (default: 1000)
- Model architecture: `model.channels` (default: [16, 32, 64, 128])

## 🏗️ Model Architecture

**BaselineCNN** - 4-stage CNN with global average pooling:

```
Input: [B, 1, 128, 128]  (grayscale images)
  ↓
Stage 1: Conv(1→16) → Conv(16→16) → MaxPool → [B, 16, 64, 64]
  ↓
Stage 2: Conv(16→32) → Conv(32→32) → MaxPool → [B, 32, 32, 32]
  ↓
Stage 3: Conv(32→64) → Conv(64→64) → MaxPool → [B, 64, 16, 16]
  ↓
Stage 4: Conv(64→128) → Conv(128→128) → MaxPool → [B, 128, 8, 8]
  ↓
Global Average Pooling → [B, 128]
  ↓
Linear(128 → 6) → [B, 6] logits
```

**Total Parameters:** ~294K

## 📊 Training Pipeline

1. **Data Loading**
   - Letterbox resize to 128×128
   - Convert to grayscale
   - Normalize to mean=0.5, std=0.5
   - DataLoader with batch_size=512, num_workers=8

2. **Training Loop** (30 epochs)
   - Forward pass through CNN
   - CrossEntropyLoss
   - Adam optimizer (lr=3e-3)
   - Save best model based on validation accuracy

3. **Evaluation**
   - Validation set evaluation after each epoch
   - Confusion matrix on best checkpoint
   - Per-class precision, recall, F1-score

4. **Outputs** (saved to `runs/<run_name>/`)
   - `checkpoints/best.pt` - Best model weights
   - `logs/metrics.csv` - Training/validation metrics
   - `logs/confusion_matrix.csv` - Confusion matrix
   - `logs/per_class_metrics.csv` - Per-class metrics

## 🔧 Key Features Added

### ✅ Device Management
- Centralized device setup (CPU/CUDA)
- Automatic device placement for all operations
- No more device mismatch errors

### ✅ Checkpoint Resume
- Resume training from previous runs
- Preserves optimizer state and best accuracy
- `--checkpoint` argument for easy resumption

### ✅ Class Distribution Debugging
- Prints class distribution for train/val sets
- Detects class imbalance issues
- Debug stats for batch statistics

### ✅ Comprehensive Metrics
- Per-epoch CSV logging
- Confusion matrix visualization
- Per-class precision/recall/F1

### ✅ GPU Optimization
- Increased batch size to 512 for better GPU utilization
- Optimized num_workers (8) for balanced I/O
- Mixed precision training ready

## 📈 Monitoring Training

During training, you'll see:
```
epoch 01/1000 | TRAIN: loss 1.7925, acc 0.167 | VAL: loss 1.7922, acc 0.167
 ->  new best: val acc 0.167
```

Check metrics in `runs/<run_name>/logs/metrics.csv`:
```csv
epoch,train_loss,train_acc,val_loss,val_acc
1,1.7925,0.167,1.7922,0.167
2,1.7924,0.167,1.7921,0.167
...
```

## 🐛 Troubleshooting

**Low accuracy (stuck at ~16.7%)**
- Enable normalization in config.yaml
- Check class distribution with debug prints
- Verify dataset is correctly loaded

**GPU not utilized**
- Increase batch_size in config.yaml
- Check with `nvidia-smi`
- Ensure CUDA is available

**Out of memory**
- Reduce batch_size
- Reduce input_size
- Reduce model channels

## 📝 Notes

- Test set evaluation is commented out by default to prevent data leakage during development
- Uncomment test evaluation in `main.py` when ready for final evaluation
- Debug image previews are saved if `debug.plot_images: true`

## 🤝 Contributing

This project was developed for ML on MCU applications with emphasis on:
- Efficient model architecture (small parameter count)
- Clean, modular codebase
- Comprehensive logging and debugging

## 📄 License

[Add your license here]
```
