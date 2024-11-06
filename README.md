# Prescription-Image-Classification-with-EfficientNet
This project focuses on building a deep learning model to classify images of prescription words using EfficientNet, achieving over 70% accuracy on validation and test sets. The code uses data augmentation techniques, strict data handling to prevent data leaks, and transfer learning for optimal results.
# Table of Contents
Project Overview
Dataset
Installation
Usage
Model Training
Results
Resources
Acknowledgments
# Project Overview
The objective was to train a prescription word classification model with an EfficientNet-B0 architecture, utilizing transfer learning. The project involved:

1. Data preprocessing and augmentation
2. Transfer learning with EfficientNet
3. Avoiding data leaks by ensuring distinct data splits
4. Achieving robust generalization on validation and test sets
# Dataset
 https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset/data
 Total image data of 4680. Among them, 60% of images are kept in training, 20% are in validation, and the rest (20%) are in testing. While splitting, data were stratified to keep equal amount of data in each class.
 Installation
Clone the repository and install the required dependencies.

git clone https://github.com/Akhila-04-03/prescription-image-classification.git
cd prescription-image-classification

Install dependencies:

pip install torch torchvision albumentations pandas pillow tqdm

Dataset Preparation: Organize the dataset as seperate folders.
Run Training: Execute the main script to train and validate the model

python train.py

Evaluate Model: After training, you can test the model on the test set by running:

python test.py
# Contact Information
Email: raveendranakhila629@gmail.com
LinkedIn: Akhila Raveendran on LinkedIn


