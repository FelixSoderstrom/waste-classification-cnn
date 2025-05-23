# Waste Classification CNN

## Project Overview
This project implements a convolutional neural network to classify waste materials into 10 categories, aiding in proper waste sorting and recycling efforts.

### Motivation
This project stems from my passion for environmental conservation, inspired by volunteer work cleaning beaches. Improper waste disposal significantly impacts our oceans and coastal environments. By creating an accurate waste classifier, this project aims to facilitate automated waste sorting and contribute to more efficient recycling processes.

## Features
- Classification of waste into 10 categories (battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash)
- Transfer learning using EfficientNet-B0 pre-trained on ImageNet
- Dynamic dataset augmentation to handle class imbalance
- Weighted loss function for unbalanced classes
- Standardized evaluation metrics

## Model Architecture
- EfficientNet-B0 backbone pre-trained on ImageNet
- Frozen weights except for the final layers (_fc, _conv_head, _bn1)
- Custom classification head:
  - Linear layer (1280 → 512)
  - ReLU activation
  - Dropout (0.3) for regularization
  - Output layer (512 → 10 classes)

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable NVIDIA GPU (strongly recommended)
- CUDA and cuDNN installed
- At least 4GB of GPU memory (8GB+ recommended)
- Sufficient disk space for the dataset (~1.5GB)

### Dataset
1. Download the [Garbage Classification Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
2. Extract it to `data_preparation/original_dataset/` directory with each category in its respective subfolder

### Installation
1. Clone this repository:
   ```bash
   git clone git@github.com:FelixSoderstrom/waste-classification-cnn.git
   cd waste-classification-cnn
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   ```bash
   python -m data_preparation.prepare_dataset
   ```
   This processes the original images, applies appropriate transformations, and creates balanced train/validation/test splits.

## Usage

### Training & Evaluation
The primary entry point for training, evaluation, and visualization is `train.py`:

```bash
python train.py
```

#### Optional Arguments
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.0005)
- `--batch_size`: Batch size (default: 32)
- `--max_epochs`: Maximum number of epochs (default: 20)
- `--num_workers`: Number of worker processes (default: 4)
- `--use_cross_validation`/`--no_cross_validation`: Enable/disable cross-validation (default: enabled)
- `--n_splits`: Number of cross-validation splits (default: 5)
- `--skip_training`: Skip training and only run evaluation
- `--model_path`: Path to model for evaluation (only used with --skip_training)
- `--skip_evaluation`: Skip evaluation and visualization (training only)

### Output
Training results, model checkpoints, and visualizations are saved in the `output/session_X` directory, where X is the session number.

## Implementation Details

### Training Pipeline
- PyTorch Lightning framework for organized training loops
- Early stopping to prevent overfitting
- Model checkpointing to save best models
- Experiment tracking with Weights & Biases
- Adam optimizer with configurable learning rate and weight decay

### Dataset Preparation
The preparation script performs several key operations:
- Resizes images to 224×224 pixels with appropriate padding
- Balances class distribution through strategic augmentation
- Ensures validation and test sets contain only non-augmented images
- Creates a standardized directory structure for training

### Evaluation & Visualization
Comprehensive evaluation tools including:
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Sample prediction visualization
- Class distribution analysis

## Challenges and Solutions

### Class Imbalance
**Problem**: The original dataset had severely imbalanced class distribution (e.g., 5326 clothes images vs. 944 battery images).
**Solution**: Implemented a two-pronged approach:
1. Applied targeted augmentation to underrepresented classes
2. Implemented weighted CrossEntropyLoss with weights inversely proportional to class frequencies

### Dataset Split Timing
**Problem**: Originally split data before augmentation, leading to imbalanced validation/test sets.
**Solution**: Modified the data preparation pipeline to:
1. Process and augment all original images first
2. Extract a fixed number of original images for validation and test sets
3. Use remaining originals plus augmented images for training


### Dataset padding
**Problem**: The dataset preparation script padded images to square sizes by using edge extrapolation. This added features in the image that does not exist. Most likely contributing to the models inability to generalize.
**Solution**: Replace the edge extrapolation and fill "empty" pixels with balck ones. Black pixels will be a part of all classes, so it will not be a feature worth learning.


### Duplicate transformations
**Problem**: The dataset was being transformed twice. Contributing to poor accuracy.
**Solution**: Remove all transformation except for the ones made in the dataset preparation script.


### Uneven class distribution
**Problem**: The dataset has alot of uneven distributions between classes. This results in a heavy model that only likes to classify clothes...
**Solution**: Underrepresented classes recieves multiple augmentation steps while overrepresented classes recieve none.
I added a custom Sampler that ensures that each class recieves the same amount of images in each epoch and rotates through different samples from larger classes with each epoch.


### Model Architecture Upgrade
**Problem**: ResNet50 was not performing well on texture-based classification tasks like waste material recognition.
**Solution**: Upgraded to EfficientNet-B0, which has been shown to perform better on texture classification while requiring fewer parameters. EfficientNet-B0 achieves better accuracy with approximately 5.3M parameters compared to ResNet50's 25.6M parameters. 