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




