
ARI3129 - AdvancedCV_Project - v1 Trashy Dataset - Preprocessed and Augmented
==============================

This dataset was exported via roboflow.com on January 13, 2025 at 12:17 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1316 images.
Trash are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 16 percent of the image
* Random rotation of between -19 and +19 degrees
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random brigthness adjustment of between -21 and +21 percent
* Random exposure adjustment of between -13 and +13 percent
* Random Gaussian blur of between 0 and 3 pixels
* Salt and pepper noise was applied to 2 percent of pixels


