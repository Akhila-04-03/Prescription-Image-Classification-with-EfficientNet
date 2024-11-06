# Prescription-Image-Classification-with-EfficientNet

This project focuses on building a deep learning model to classify images of prescription words using EfficientNet, achieving over 70% accuracy on validation and test sets. The code utilizes data augmentation techniques, strict data handling to prevent data leaks, and transfer learning for optimal results.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Acknowledgments](#acknowledgments)
- [Contact Information](#contact-information)

## Project Overview

The objective of this project was to train a prescription word classification model using the EfficientNet-B0 architecture, leveraging transfer learning. Key steps involved in the project:

1. **Data Preprocessing and Augmentation**
2. **Transfer Learning with EfficientNet**
3. **Avoiding Data Leaks** by ensuring distinct data splits
4. Achieving robust generalization on **validation** and **test sets**

## Dataset

The dataset used for this project is the "Doctor's Handwritten Prescription BD" dataset, available on Kaggle:

- [Doctor's Handwritten Prescription BD Dataset](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset/data)

### Dataset Details:

- Total images: **4680**
- Dataset split:
  - **60%** for training
  - **20%** for validation
  - **20%** for testing
- Stratified split to ensure each class is represented equally across training, validation, and test sets.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Akhila-04-03/prescription-image-classification.git
cd prescription-image-classification

pip install torch torchvision albumentations pandas pillow tqdm

## Dataset Preperation
/dataset
    /train
        /class_1
        /class_2
        ...
    /val
        /class_1
        /class_2
        ...
    /test
        /class_1
        /class_2
        ...
## Model Training

python train.py


## Evaluate the model

python test.py
----


### Acknowledgments
EfficientNet for its powerful transfer learning capabilities.
Kaggle for providing the dataset: "Doctor's Handwritten Prescription BD Dataset".
PyTorch (https://pytorch.org/) for deep learning frameworks.
Torchvision (https://pytorch.org/vision/stable/index.html) for image processing.
Albumentations (https://albumentations.ai/) for data augmentation and image transformations.

## Contact Information
Email: raveendranakhila629@gmail.com
LinkedIn: Akhila Raveendran on LinkedIn

### What Changed:
1. **Reordering of sections**: The **Acknowledgments** and **Contact Information** sections were moved to the bottom of the README (after the "Evaluate the Model" section).
2. **Corrected Table of Contents**: I made sure the **Table of Contents** links match the correct sections.

Now, the **Acknowledgments** and **Contact Information** are placed properly at the end of the README file, after the sections related to the training and testing code.

This should fix the issue, and the sections will no longer appear under the training/testing sections. Let me know if you need further adjustments!
