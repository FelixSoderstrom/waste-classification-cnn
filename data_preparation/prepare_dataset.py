#!/usr/bin/env python

import os
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

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
TARGET_VAL_SIZE = 200  # 200 images per class for validation
TARGET_TEST_SIZE = 200  # 200 images per class for test
# We're no longer using these ratios:
# TRAIN_RATIO = 0.7
# VAL_RATIO = 0.15
# TEST_RATIO = 0.15

# Define category sizes for augmentation strategy
OVER_REPRESENTED = {"clothes", "glass"}  # No augmentation
UNDER_REPRESENTED = {
    "battery",
    "biological",
    "metal",
    "trash",
}  # Full augmentation
# Middle classes get partial augmentation automatically


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

# Individual augmentations
affine_transform = transforms.Compose(
    [
        base_transform,
        transforms.RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.05)
        ),
    ]
)

elastic_transform = transforms.Compose(
    [
        base_transform,
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
    ]
)

color_jitter_transform = transforms.Compose(
    [
        base_transform,
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
)


def create_directories():
    """Create required directories for train, validation, and test splits."""
    for split in ["training", "validation", "test"]:
        for category in CATEGORIES:
            dir_path = PROCESSED_DATASET_PATH / split / category
            dir_path.mkdir(parents=True, exist_ok=True)

    print("Created directories for processed dataset")


def prepare_balanced_dataset():
    """
    Prepare a balanced dataset with fixed numbers of images per class for test and validation:
    1. Apply base transformation to ALL original images
    2. Create augmented versions based on category strategy
    3. Extract fixed number of original images per class for test and validation
    4. Use remaining original images + all augmented images for training
    """
    # Data structures to store processed images
    basic_images = {category: [] for category in CATEGORIES}
    augmented_images = {category: [] for category in CATEGORIES}
    category_stats = {
        category: {"original_count": 0, "augmented_count": 0}
        for category in CATEGORIES
    }

    # Create necessary directories
    create_directories()

    # Step 1 & 2: Process and augment all original images
    for category in CATEGORIES:
        print(f"Processing {category} images...")
        source_dir = ORIGINAL_DATASET_PATH / category

        # Get all image files
        image_files = [
            f
            for f in os.listdir(source_dir)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]

        # Record original count
        category_stats[category]["original_count"] = len(image_files)

        if not image_files:
            print(f"No images found in {source_dir}")
            continue

        # Process each original image
        for filename in tqdm(
            image_files, desc=f"Processing {category}", leave=False
        ):
            image_path = source_dir / filename
            try:
                # Load and apply base transform
                image = Image.open(image_path).convert("RGB")
                base_image = base_transform(image)

                # Store base transformed image
                basic_images[category].append(
                    {"filename": filename, "image": base_image}
                )

                # Determine augmentation strategy based on category
                apply_affine = category not in OVER_REPRESENTED
                apply_elastic = category in UNDER_REPRESENTED
                apply_color_jitter = category in UNDER_REPRESENTED

                # For middle categories, use existing logic
                if (
                    category not in OVER_REPRESENTED
                    and category not in UNDER_REPRESENTED
                ):
                    if len(image_files) < 1400:
                        apply_elastic = True
                        apply_color_jitter = False
                    elif len(image_files) < 1700:
                        apply_elastic = False
                        apply_color_jitter = False
                    else:
                        apply_elastic = False
                        apply_color_jitter = False

                # Apply and store augmentations
                name_parts = os.path.splitext(filename)

                if apply_affine:
                    aug_image = affine_transform(image)
                    aug_filename = f"{name_parts[0]}_affine{name_parts[1]}"
                    augmented_images[category].append(
                        {"filename": aug_filename, "image": aug_image}
                    )
                    category_stats[category]["augmented_count"] += 1

                if apply_elastic:
                    aug_image = elastic_transform(image)
                    aug_filename = f"{name_parts[0]}_elastic{name_parts[1]}"
                    augmented_images[category].append(
                        {"filename": aug_filename, "image": aug_image}
                    )
                    category_stats[category]["augmented_count"] += 1

                if apply_color_jitter:
                    aug_image = color_jitter_transform(image)
                    aug_filename = f"{name_parts[0]}_color{name_parts[1]}"
                    augmented_images[category].append(
                        {"filename": aug_filename, "image": aug_image}
                    )
                    category_stats[category]["augmented_count"] += 1

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # Step 3-7: Distribute images to final directories
    distribution_counts = {
        category: {"training": 0, "validation": 0, "test": 0}
        for category in CATEGORIES
    }

    for category in CATEGORIES:
        print(f"Distributing {category} images to splits...")

        # Shuffle the basic images for random selection
        random.shuffle(basic_images[category])

        basic_count = len(basic_images[category])

        # Define how many images to extract for validation and test
        # If we don't have enough images, take what we can
        test_count = min(TARGET_TEST_SIZE, basic_count // 3)
        val_count = min(TARGET_VAL_SIZE, (basic_count - test_count) // 2)

        # Extract images for test
        test_dir = PROCESSED_DATASET_PATH / "test" / category
        for i in range(test_count):
            if not basic_images[category]:  # Check if we've run out
                break
            img_data = basic_images[category].pop()  # Take from the end
            img_data["image"].save(
                test_dir / img_data["filename"], quality=95
            )
            distribution_counts[category]["test"] += 1

        # Extract images for validation
        val_dir = PROCESSED_DATASET_PATH / "validation" / category
        for i in range(val_count):
            if not basic_images[category]:  # Check if we've run out
                break
            img_data = basic_images[category].pop()  # Take from the end
            img_data["image"].save(val_dir / img_data["filename"], quality=95)
            distribution_counts[category]["validation"] += 1

        # Save remaining basic images to training
        train_dir = PROCESSED_DATASET_PATH / "training" / category
        for img_data in basic_images[category]:
            img_data["image"].save(
                train_dir / img_data["filename"], quality=95
            )
            distribution_counts[category]["training"] += 1

        # Save all augmented images to training
        for img_data in augmented_images[category]:
            img_data["image"].save(
                train_dir / img_data["filename"], quality=95
            )
            distribution_counts[category]["training"] += 1

    return distribution_counts, category_stats


def print_dataset_summary(distribution_counts, category_stats):
    """Print a summary of the dataset preparation results."""
    print("\nBalanced Dataset Preparation Summary:")
    print("-" * 80)
    print(
        f"{'Category':<15} {'Training':<10} {'Validation':<10} {'Test':<10} {'Orig Total':<12} {'Aug Count':<10} {'Total':<10}"
    )
    print("-" * 80)

    total_train, total_val, total_test, total_orig, total_aug, total_all = (
        0,
        0,
        0,
        0,
        0,
        0,
    )

    for category in CATEGORIES:
        train_count = distribution_counts[category]["training"]
        val_count = distribution_counts[category]["validation"]
        test_count = distribution_counts[category]["test"]
        orig_count = category_stats[category]["original_count"]
        aug_count = category_stats[category]["augmented_count"]
        total_count = train_count + val_count + test_count

        print(
            f"{category:<15} {train_count:<10} {val_count:<10} {test_count:<10} {orig_count:<12} {aug_count:<10} {total_count:<10}"
        )

        total_train += train_count
        total_val += val_count
        total_test += test_count
        total_orig += orig_count
        total_aug += aug_count
        total_all += total_count

    print("-" * 80)
    print(
        f"{'Total':<15} {total_train:<10} {total_val:<10} {total_test:<10} {total_orig:<12} {total_aug:<10} {total_all:<10}"
    )

    print("\nClass Balance Analysis:")
    print(f"- Target for validation: {TARGET_VAL_SIZE} images per class")
    print(f"- Target for test: {TARGET_TEST_SIZE} images per class")
    print(
        f"- Over-represented categories (no augmentation): {', '.join(OVER_REPRESENTED)}"
    )
    print(
        f"- Under-represented categories (full augmentation): {', '.join(UNDER_REPRESENTED)}"
    )

    print("\nNote: All augmented images are in the training set only.")
    print(
        "      Validation and test sets contain only original (non-augmented) images."
    )


def main():
    """Main function to prepare the balanced dataset."""
    print("Starting balanced dataset preparation...")

    # Process dataset with balanced approach
    distribution_counts, category_stats = prepare_balanced_dataset()

    # Print summary
    print_dataset_summary(distribution_counts, category_stats)

    print("\nBalanced dataset preparation complete!")


if __name__ == "__main__":
    main()
