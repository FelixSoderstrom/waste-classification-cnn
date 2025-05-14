#!/usr/bin/env python

import os
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define constants
ORIGINAL_DATASET_PATH = Path("data_preparation/original_dataset")
PROCESSED_DATASET_PATH = Path("src/dataset")
CATEGORIES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]
TARGET_SIZE = (224, 224)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Ensure the ratio sums to 1
assert math.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0)


# Define transforms for different splits
# Padding transformation to maintain aspect ratio
class PadToSize:
    """Pad image to target size with background filled by edge pixels."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        target_w, target_h = self.size

        # Calculate padding
        padding_w = max(0, target_w - w)
        padding_h = max(0, target_h - h)
        padding_left = padding_w // 2
        padding_right = padding_w - padding_left
        padding_top = padding_h // 2
        padding_bottom = padding_h - padding_top

        # Apply padding
        return F.pad(
            img,
            (padding_left, padding_top, padding_right, padding_bottom),
            padding_mode="edge",
        )


# Base transform for all splits (resize with padding to maintain aspect ratio)
base_transform = transforms.Compose(
    [
        PadToSize(TARGET_SIZE),
        transforms.Resize(
            TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC
        ),
    ]
)

# Training transform (includes augmentation)
train_transform = transforms.Compose(
    [
        base_transform,
        transforms.RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.05)
        ),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
)

# Validation and test transform (only resizing)
val_test_transform = base_transform


def create_directories():
    """Create required directories for train, validation, and test splits."""
    for split in ["training", "validation", "test"]:
        for category in CATEGORIES:
            dir_path = PROCESSED_DATASET_PATH / split / category
            dir_path.mkdir(parents=True, exist_ok=True)

    print("Created directories for processed dataset")


def process_images():
    """Process images by category, resizing and splitting into appropriate sets."""
    category_counts: Dict[str, Dict[str, int]] = {}

    for category in CATEGORIES:
        print(f"Processing {category} images...")
        source_dir = ORIGINAL_DATASET_PATH / category

        # Get all image files
        image_files = [
            f
            for f in os.listdir(source_dir)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]
        total_images = len(image_files)

        if total_images == 0:
            print(f"No images found in {source_dir}")
            continue

        # Shuffle image files to ensure random distribution
        random.shuffle(image_files)

        # Calculate split sizes
        train_size = int(total_images * TRAIN_RATIO)
        val_size = int(total_images * VAL_RATIO)
        # Remainder goes to test to ensure we use all images
        test_size = total_images - train_size - val_size

        # Split image files
        train_files = image_files[:train_size]
        val_files = image_files[train_size : train_size + val_size]
        test_files = image_files[train_size + val_size :]

        # Process images for each split
        train_processed = process_split_images(
            source_dir, "training", train_files, train_transform, category
        )
        process_split_images(
            source_dir, "validation", val_files, val_test_transform, category
        )
        process_split_images(
            source_dir, "test", test_files, val_test_transform, category
        )

        # Store counts for summary - for training we have both original and augmented versions
        category_counts[category] = {
            "training": train_processed,  # This now includes both original and augmented
            "validation": len(val_files),
            "test": len(test_files),
            "total": total_images,
            "processed_total": train_processed
            + len(val_files)
            + len(test_files),
        }

    return category_counts


def process_split_images(
    source_dir: Path, split: str, files: List[str], transform, category: str
) -> int:
    """
    Process images for a specific split using the provided transform.

    Returns:
        int: Number of processed images saved
    """
    target_dir = PROCESSED_DATASET_PATH / split / category
    processed_count = 0

    for filename in tqdm(
        files, desc=f"{split.capitalize()} - {category}", leave=False
    ):
        # Load image
        image_path = source_dir / filename
        try:
            image = Image.open(image_path).convert("RGB")

            if split == "training":
                # For training set, save both original (resized) and augmented versions

                # 1. Save original with just resizing/padding
                base_image = base_transform(image)
                base_path = target_dir / filename
                base_image.save(base_path, quality=95)
                processed_count += 1

                # 2. Save augmented version with modified filename
                augmented_image = train_transform(image)
                # Get filename parts to add suffix
                name_parts = os.path.splitext(filename)
                aug_filename = f"{name_parts[0]}_augmented{name_parts[1]}"
                aug_path = target_dir / aug_filename
                augmented_image.save(aug_path, quality=95)
                processed_count += 1
            else:
                # For validation and test sets, just apply base transform
                transformed_image = transform(image)
                target_path = target_dir / filename
                transformed_image.save(target_path, quality=95)
                processed_count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return processed_count


def main():
    """Main function to prepare the dataset."""
    print("Starting dataset preparation...")

    # Create directories
    create_directories()

    # Process images
    category_counts = process_images()

    # Print summary
    print("\nDataset Preparation Summary:")
    print("-" * 80)
    print(
        f"{'Category':<15} {'Training':<10} {'Validation':<10} {'Test':<10} {'Orig Total':<12} {'Processed':<10}"
    )
    print("-" * 80)

    total_train, total_val, total_test, total_all, total_processed = (
        0,
        0,
        0,
        0,
        0,
    )

    for category, counts in category_counts.items():
        print(
            f"{category:<15} {counts['training']:<10} {counts['validation']:<10} {counts['test']:<10} {counts['total']:<12} {counts['processed_total']:<10}"
        )
        total_train += counts["training"]
        total_val += counts["validation"]
        total_test += counts["test"]
        total_all += counts["total"]
        total_processed += counts["processed_total"]

    print("-" * 80)
    print(
        f"{'Total':<15} {total_train:<10} {total_val:<10} {total_test:<10} {total_all:<12} {total_processed:<10}"
    )

    # Verify split ratios
    if total_all > 0:
        # Original images split ratio (before augmentation)
        orig_train_pct = (
            (total_train / 2) / total_all * 100 if total_train > 0 else 0
        )
        orig_val_pct = total_val / total_all * 100
        orig_test_pct = total_test / total_all * 100
        print(
            f"\nOriginal images split ratios: Training={orig_train_pct:.1f}%, Validation={orig_val_pct:.1f}%, Test={orig_test_pct:.1f}%"
        )
        print(
            f"Target split ratios: Training={TRAIN_RATIO*100:.1f}%, Validation={VAL_RATIO*100:.1f}%, Test={TEST_RATIO*100:.1f}%"
        )

        # Note about augmented images
        print(
            f"\nNote: Training set includes both original and augmented versions of each image."
        )
        print(
            f"      Total processed images: {total_processed} (includes {total_train//2 if total_train > 0 else 0} augmented images)"
        )

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
