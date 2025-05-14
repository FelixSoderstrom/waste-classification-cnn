# Waste classifier by Felix Söderström

## Project description

I want to build a convolutional nerual network to classify waste materials.
The plan is to transfer learn the ResNet50 with [this dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2).


## Data

### Preparing the dataset

The original dataset came organized into subfolders of materials.
Each image was of different resolution and since I know ResNet50 requires specific image resolutions I wrote a script that:
- Takes images from '/data_preparation/original_dataset/{subcategory}'
- Resizes images accordingly
- Pads images to correct aspect ratio
- Divides each category into training, validation and test sets (70/15/15 split)
- Augments training images
- Saves all images into '/src/dataset/{train/val/test}/{subcategory}'

I can now start using the images for training.

## Model Architecture

The model uses transfer learning with a ResNet50 backbone pre-trained on ImageNet. Key aspects of the architecture include:

- ResNet50 base with frozen weights except for the final layer
- Custom classification head with:
  - Linear layer (2048 → 512)
  - ReLU activation
  - Dropout (0.3)
  - Output layer (512 → 11 classes)
- The model classifies waste into 10 categories (battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash) plus an "other" category

## Training Process

The training pipeline is implemented using PyTorch Lightning with the following features:

- Uses pre-augmented images (224×224 pixels) from the dataset preparation step
- Includes options for both standard training and k-fold cross-validation
- Implements early stopping to prevent overfitting
- Tracks experiments using Weights & Biases logging
- Uses Adam optimizer with configurable learning rate and weight decay

**⚠️ IMPORTANT:** This model requires GPU acceleration for effective training. The code automatically detects and uses CUDA-capable GPUs if available. Training on CPU is not recommended due to the computationally intensive nature of the ResNet50 architecture.

## Evaluation & Visualization

The project includes comprehensive evaluation and visualization tools:

- Evaluation metrics including accuracy, precision, recall, and F1-score
- Confusion matrix visualization to identify misclassifications
- Sample prediction visualization showing model predictions on test images
- Class distribution plots across training, validation, and test sets

## Usage Instructions

### Training the model

```bash
python src/training/training.py
```

Optional arguments:
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.0005)
- `--batch_size`: Batch size (default: 32)
- `--max_epochs`: Maximum number of epochs (default: 10)
- `--use_cross_validation`/`--no_cross_validation`: Enable/disable cross-validation
- `--n_splits`: Number of cross-validation splits (default: 5)

### Evaluating the model

```bash
python src/evaluation/evaluation.py
```

Optional arguments:
- `--model_path`: Path to the trained model checkpoint
- `--batch_size`: Batch size for evaluation

### Visualizing model performance

```bash
python src/evaluation/visualization.py
```

Optional arguments:
- `--model_path`: Path to the trained model checkpoint

## System Requirements

- CUDA-capable NVIDIA GPU (strongly recommended)
- CUDA and cuDNN installed
- At least 4GB of GPU memory (8GB+ recommended)
- Sufficient disk space for the dataset

## Software Requirements

The project requires the following Python packages:
- torch >= 2.1.0
- torchvision >= 0.16.0
- pytorch-lightning >= 2.0.0
- wandb >= 0.15.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.2.0
- numpy >= 1.24.0
- Pillow >= 10.0.0
- tqdm >= 4.66.0




