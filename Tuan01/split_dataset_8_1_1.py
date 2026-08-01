import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_caption_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "image" in df.columns and "caption" in df.columns:
        return df[["image", "caption"]].copy()

    if len(df.columns) < 2:
        raise ValueError("Caption file must have at least two columns: image and caption.")

    df = df.iloc[:, :2].copy()
    df.columns = ["image", "caption"]
    return df


def split_items(items: list[str], train_ratio: float, val_ratio: float) -> tuple[list[str], list[str], list[str]]:
    train_count = round(len(items) * train_ratio)
    val_count = round(len(items) * val_ratio)

    train_items = items[:train_count]
    val_items = items[train_count : train_count + val_count]
    test_items = items[train_count + val_count :]

    return train_items, val_items, test_items


def save_split(df: pd.DataFrame, image_names: list[str], output_path: Path) -> pd.DataFrame:
    split_df = df[df["image"].isin(image_names)].copy()
    split_df.to_csv(output_path, index=False)
    return split_df


def copy_split_images(image_names: list[str], image_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = []

    for image_name in image_names:
        source = image_dir / image_name
        if source.exists():
            shutil.copy2(source, output_dir / image_name)
        else:
            missing.append(image_name)

    if missing:
        print(f"[!] Missing {len(missing)} images in {image_dir}. First missing file: {missing[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a Flickr8k-style caption dataset into train/val/test.")
    parser.add_argument("--captions", required=True, help="Path to captions.txt/csv with image,caption columns.")
    parser.add_argument("--image-dir", help="Path to the image folder. Required only with --copy-images.")
    parser.add_argument("--output-dir", default="splits", help="Directory for generated split CSV files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for a reproducible split.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--copy-images", action="store_true", help="Also copy images into train/val/test folders.")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}.")

    captions_path = Path(args.captions)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None

    if args.copy_images and image_dir is None:
        raise ValueError("--image-dir is required when using --copy-images.")

    df = normalize_caption_columns(pd.read_csv(captions_path))
    image_names = sorted(df["image"].astype(str).unique().tolist())
    random.Random(args.seed).shuffle(image_names)

    train_images, val_images, test_images = split_items(image_names, args.train_ratio, args.val_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = save_split(df, train_images, output_dir / "train_captions.csv")
    val_df = save_split(df, val_images, output_dir / "val_captions.csv")
    test_df = save_split(df, test_images, output_dir / "test_captions.csv")

    if args.copy_images and image_dir is not None:
        copy_split_images(train_images, image_dir, output_dir / "train" / "Images")
        copy_split_images(val_images, image_dir, output_dir / "val" / "Images")
        copy_split_images(test_images, image_dir, output_dir / "test" / "Images")

    print("[*] Split complete")
    print(f"    train: {len(train_images)} images, {len(train_df)} captions")
    print(f"    val:   {len(val_images)} images, {len(val_df)} captions")
    print(f"    test:  {len(test_images)} images, {len(test_df)} captions")
    print(f"[*] CSV files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
