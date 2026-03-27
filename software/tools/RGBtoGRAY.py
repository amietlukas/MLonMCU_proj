"""
Convert an image dataset from RGB to grayscale (uint8 0..255) while preserving the
exact folder structure.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def convert_image_to_gray(src: Path, dst: Path, *, jpg_quality: int = 95) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            gray = im.convert("L")  # 8-bit grayscale, values 0..255
            suffix = src.suffix.lower()
            save_kwargs = {}
            if suffix in {".jpg", ".jpeg"}:
                save_kwargs.update(dict(quality=jpg_quality, optimize=True))
            gray.save(dst, **save_kwargs)
    except (UnidentifiedImageError, OSError) as e:
        raise RuntimeError(f"Failed to read/convert image: {src} ({e})") from e


def main() -> None:
    script_dir = Path(__file__).resolve().parent  # .../MLonMCU/src
    project_root = script_dir.parent              # .../MLonMCU

    default_in = project_root / "datasets" / "HAGRID" / "hagrid_0_1_classification"
    default_out = project_root / "datasets" / "HAGRID" / "hagrid_0_1_classification_gray"

    parser = argparse.ArgumentParser(
        description="Convert an RGB dataset to grayscale while preserving structure."
    )
    parser.add_argument("--in_root", type=Path, default=default_in, help="Input dataset root folder.")
    parser.add_argument("--out_root", type=Path, default=default_out, help="Output dataset root folder.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing out_root (still overwrites files).",
    )
    parser.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality for saved grayscale JPEGs (1..100).")

    args = parser.parse_args()

    in_root: Path = args.in_root.resolve()
    out_root: Path = args.out_root.resolve()

    if not in_root.exists() or not in_root.is_dir():
        raise SystemExit(f"[ERROR] Input root does not exist or is not a directory: {in_root}")

    if out_root.exists() and not args.overwrite:
        raise SystemExit(
            f"[ERROR] Output root already exists: {out_root}\n"
            f"        Use --overwrite or choose a different --out_root."
        )

    out_root.mkdir(parents=True, exist_ok=True)

    total = converted = copied = failed = 0

    for src in iter_files(in_root):
        total += 1
        rel = src.relative_to(in_root)
        dst = out_root / rel

        if src.suffix.lower() in IMAGE_EXTS:
            try:
                convert_image_to_gray(src, dst, jpg_quality=args.jpg_quality)
                converted += 1
            except RuntimeError as e:
                failed += 1
                print(f"[WARN] {e}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

        if total % 500 == 0:
            print(f"[INFO] Processed {total} files... (converted={converted}, copied={copied}, failed={failed})")

    print("\n[DONE]")
    print(f"  Input : {in_root}")
    print(f"  Output: {out_root}")
    print(f"  Total files      : {total}")
    print(f"  Images converted : {converted}")
    print(f"  Non-images copied: {copied}")
    print(f"  Failed           : {failed}")


if __name__ == "__main__":
    main()
