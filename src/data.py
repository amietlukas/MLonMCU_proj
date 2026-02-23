from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# IMPORTANT: explicit order
CLASS_NAMES: List[str] = [
    "palm",    # STOP                           0
    "rock",    # DRIVE F STRAIGHT               1
    "pinkie",  # STEER F RIGHT                  2
    "one",     # STEER F LEFT                   3
    "fist",    # DRIVE B STRAIGHT               4
    "others",  # OTHER / background class       5
]


CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}


def scan_split(dataset_root: Path, split: str) -> List[Tuple[Path, int]]:
    """
    Collect all image paths for a given split and map them to label indices.
    Returns a flat list of (image_path, y).
    samples = [ (Path(".../train/palm/img1.jpg"), 0),... , (Path(".../train/rock/img2.jpg"), 1),... ]
    """

    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    samples: List[Tuple[Path, int]] = []

    # iterate in explicit class order
    for cls_name in CLASS_NAMES:
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder missing: {cls_dir}")

        # gather image files
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                samples.append((p, CLASS_TO_IDX[cls_name]))

    if len(samples) == 0:
        raise RuntimeError(f"No images found in {split_dir}")

    return samples



# =========== Transform ===========
# Letterbox class
class LetterboxSquare:
    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = self.size / max(w, h)
        nw, nh = int(round(w * s)), int(round(h * s))

        img = F.resize(img, (nh, nw), interpolation=F.InterpolationMode.BILINEAR)

        pad_w = self.size - nw
        pad_h = self.size - nh
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = F.pad(img, (left, top, right, bottom), fill=self.fill)

        # check if size is correct
        if img.size != (self.size, self.size):
            raise RuntimeError(f"LetterboxSquare produced {img.size}, expected {(self.size, self.size)}")

        return img

# apply the transform
def build_transform(cfg: Dict[str, Any], train: bool) -> T.Compose:
    # with argument "train" we can later add data augmentation for training set

    # look up config for transform parameters
    size = int(cfg["data"]["input_size"])
    pp = cfg.get("preprocess", {})
    pp_type = pp.get("type", "letterbox")
    fill = int(pp.get("fill", 0))
    norm = pp.get("normalize", {})
    mean = float(norm.get("mean", 0.5))
    std = float(norm.get("std", 0.5))
    in_channels = int(cfg["model"]["in_channels"])

    if pp_type == "letterbox":
        spatial = LetterboxSquare(size=size, fill=fill) # callable object 
    else:
        raise ValueError(f"Unknown preprocess.type: {pp_type}")

    # tf = T.Compose([
    #     T.Grayscale(num_output_channels=1),
    #     spatial,
    #     T.ToTensor(), # [0,1], shape [1,H,W]
    #     T.Normalize(mean=[mean], std=[std]),
    # ])
    # return  tf # callable transform object

    # channel handling + normalization
    if in_channels == 1:
        channel_tf = [T.Grayscale(num_output_channels=1)]
        norm_mean = [mean]
        norm_std  = [std]
    elif in_channels == 3:
        channel_tf = []  # already RGB
        norm_mean = [mean, mean, mean]
        norm_std  = [std, std, std]
    else:
        raise ValueError(f"Unsupported input_channels: {in_channels}")

    tf = T.Compose( # expects one list
        channel_tf + [
            spatial,
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
    return tf

# =========== Dataset ===========
class HagridGestureDataset(Dataset):

    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # return: (x: image, y: label, img_path)

        img_path, y = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform is None:
            raise RuntimeError("No transform provided. Please provide a transform for image preprocessing.")

        # preprocessing
        x = self.transform(img)

        return x, torch.tensor(y, dtype=torch.long), str(img_path)

def build_datasets(cfg: Dict[str, Any]):
    dataset_root: Path = cfg["data"]["dataset_root"]

    train_samples = scan_split(dataset_root, "train")
    val_samples   = scan_split(dataset_root, "val")
    test_samples  = scan_split(dataset_root, "test")

    tf_train = build_transform(cfg, train=True)
    tf_eval  = build_transform(cfg, train=False)

    train_ds = HagridGestureDataset(train_samples, transform=tf_train)
    val_ds   = HagridGestureDataset(val_samples, transform=tf_eval)
    test_ds  = HagridGestureDataset(test_samples, transform=tf_eval)

    return train_ds, val_ds, test_ds



# =========== Dataloaders ===========
def build_dataloaders(cfg: Dict[str, Any]):
    train_ds, val_ds, test_ds = build_datasets(cfg)

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    shuffle = bool(cfg["train"]["shuffle"])

    persistent = nw > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        persistent_workers=persistent,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        persistent_workers=persistent,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        persistent_workers=persistent,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader