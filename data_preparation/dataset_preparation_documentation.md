# Waste Classification Dataset Preparation

This document explains the functionality and implementation details of the `prepare_dataset.py` script, which is designed to prepare a waste classification image dataset for use in a convolutional neural network (CNN).

## How to use

1. Make sure you have downloaded the original dataset and put it in the `data_preparation/original_dataset` folder. The material-subfolders should be directly in 'original_dataset'
2. Make sure you have created a venv and activated it.
```bash
python -m venv venv  # Make sure this is Python version 3.10
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Then run the script with
```bash
python -m data_preparation.prepare_dataset
```




## Overview

The script takes raw waste classification images from multiple categories, processes them (resizing and applying data augmentation), and organizes them into structured training, validation, and test sets. This preparation is a critical step in the machine learning workflow, ensuring that the model has high-quality, consistent data for training and evaluation.

## Directory Structure

### Input Structure

Original images are organized by category in the following structure:
```
/data_preparation/original_dataset/
    ├── battery/
    ├── biological/
    ├── cardboard/
    ├── clothes/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    ├── shoes/
    └── trash/
```

### Output Structure

Processed images are organized into training, validation, and test sets:
```
/src/dataset/
    ├── training/
    │   ├── battery/
    │   │   ├── image1.jpg                 # Resized original
    │   │   ├── image1_augmented.jpg       # Augmented version
    │   │   ├── image2.jpg
    │   │   └── image2_augmented.jpg
    │   ├── biological/
    │   └── ... (all categories)
    ├── validation/
    │   ├── battery/
    │   ├── biological/
    │   └── ... (all categories)
    └── test/
        ├── battery/
        ├── biological/
        └── ... (all categories)
```

## Workflow

The script follows a systematic approach to prepare the dataset:

1. **Directory Creation**: Creates all necessary directories in the target location
2. **Category Processing**: Processes each waste category sequentially
3. **Image Processing**: For each category:
   - Lists all images in the source directory
   - Shuffles images randomly (using seed 42 for reproducibility)
   - Splits images into training (70%), validation (15%), and test (15%) sets
   - Applies appropriate transformations to each image
   - For training images: creates both a resized original and an augmented version
   - Saves processed images to their target locations
4. **Summary Generation**: Creates a detailed summary of the processing results

## Image Transformations

### Base Transformation

All images (training, validation, and test) undergo the following base transformations:

1. **Padding**: Custom `PadToSize` class adds padding to maintain aspect ratio
2. **Resizing**: High-quality resizing to 224×224 pixels using bicubic interpolation

```python
base_transform = transforms.Compose([
    PadToSize(TARGET_SIZE),
    transforms.Resize(
        TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC
    ),
])
```

### Custom Padding

The custom `PadToSize` class calculates appropriate padding to maintain the original aspect ratio of images while ensuring they fit within the target size. It uses edge-pixel padding to create a natural-looking border that preserves the image's visual integrity.

### Training Set Augmentation

For the training set, two versions of each image are created:

1. **Original Version**: Just resized with padding to maintain aspect ratio
2. **Augmented Version**: Additional transformations applied to increase diversity:

   - **Random Affine Transformations**:
     - Rotation: ±10 degrees
     - Translation: ±10% horizontally and vertically
     - Scale: 85% to 105% of original size

   - **Elastic Transformations**:
     - Alpha: 50.0 (displacement field amplitude)
     - Sigma: 5.0 (smoothing factor for the displacement field)

   - **Color Jitter**:
     - Brightness: ±20%
     - Contrast: ±20%

```python
train_transform = transforms.Compose([
    base_transform,
    transforms.RandomAffine(
        degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.05)
    ),
    transforms.ElasticTransform(alpha=50.0, sigma=5.0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])
```

## Dataset Splitting Methodology

The script splits each category's images using these steps:

1. Images are randomly shuffled using a fixed random seed (42) for reproducibility
2. The first 70% of images are assigned to the training set
3. The next 15% are assigned to the validation set
4. The final 15% are assigned to the test set

This stratified splitting approach ensures that each category maintains the same ratio across all three sets, which is important for balanced model training and evaluation.

## Naming Convention

To distinguish between original and augmented images in the training set:
- Original images keep their original filenames
- Augmented versions have "_augmented" added before the file extension
  - Example: "waste_item.jpg" → "waste_item_augmented.jpg"

## Output Summary

Upon completion, the script generates a comprehensive summary that includes:

1. **Per-Category Statistics**: 
   - Number of images in each split for each category
   - Number of original images vs. processed images (including augmented versions)

2. **Total Statistics**: 
   - Total number of images across all categories and splits
   - Total number of processed images (including augmentations)

3. **Split Ratio Verification**: 
   - Comparison of actual vs. target split ratios for original images

Example summary output:
```
Dataset Preparation Summary:
--------------------------------------------------------------------------------
Category        Training    Validation  Test       Orig Total   Processed  
--------------------------------------------------------------------------------
battery         140         15          15          70            170       
biological      210         23          22          105           255       
...
--------------------------------------------------------------------------------
Total           1400        150         150         700           1700      

Original images split ratios: Training=70.0%, Validation=15.0%, Test=15.0%
Target split ratios: Training=70.0%, Validation=15.0%, Test=15.0%

Note: Training set includes both original and augmented versions of each image.
      Total processed images: 1700 (includes 700 augmented images)
```

## Requirements

The script requires the following Python packages:
- torch>=2.1.0
- torchvision>=0.16.0
- Pillow>=10.0.0
- tqdm>=4.66.0

## Error Handling

The script includes robust error handling to:
- Skip empty categories
- Catch and report individual image processing errors
- Verify directory structures
- Validate split ratios

## Conclusion

This dataset preparation script ensures that your waste classification CNN has high-quality, consistently processed image data, properly organized into appropriate training, validation, and test sets. The training set now contains both original (resized) images and their augmented counterparts, providing a richer training dataset while preserving the original content. This approach allows models to learn from both standard and augmented examples, potentially improving generalization and robustness. 