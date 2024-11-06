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
```bash
Clone the repository and install the required dependencies:

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
It sounds like there might be a structural issue with the Markdown syntax, such as missing line breaks or unclosed sections, which causes the content after the **Installation** section to be treated as part of that section.

### Possible Causes:
1. **Missing line breaks**: In Markdown, you need to ensure there is a blank line between different sections or paragraphs. If there's no blank line between the content, it can cause everything after a section (e.g., Installation) to be treated as part of that section.
2. **Incorrect heading formatting**: Make sure each section (like **Acknowledgments** and **Contact Information**) has a valid heading and is clearly separated from the previous section.
3. **Improper indentation or list syntax**: Sometimes improper indentation in lists or other elements can cause unexpected behavior in Markdown rendering.

### Fixing the Issue:

Here’s how you can ensure proper structure:

1. **Add blank lines between sections**: After every section or heading, ensure that there is a blank line before starting a new section.
2. **Ensure correct Markdown syntax**: Each major section should have a corresponding heading, and sections like **Acknowledgments** and **Contact Information** should be clearly separated.

### Corrected `README.md` with Fixes:

```markdown
# Prescription-Image-Classification-with-EfficientNet

This project focuses on building a deep learning model to classify images of prescription words using EfficientNet, achieving over 70% accuracy on validation and test sets. The code utilizes data augmentation techniques, strict data handling to prevent data leaks, and transfer learning for optimal results.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluate the Model](#evaluate-the-model)
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
```

### Dataset Preparation

Make sure your dataset is organized into separate folders for training, validation, and testing:

```
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
```

## Model Training

To train the model, execute the following command:

```bash
python train.py
```

## Evaluate the Model

After training, you can evaluate the model on the test set by running:

```bash
python test.py
```

---

## Acknowledgments

- **EfficientNet** for its powerful transfer learning capabilities.
- **Kaggle** for providing the dataset: "Doctor's Handwritten Prescription BD Dataset".
- **PyTorch** (https://pytorch.org/) for deep learning frameworks.
- **Torchvision** (https://pytorch.org/vision/stable/index.html) for image processing.
- **Albumentations** (https://albumentations.ai/) for data augmentation and image transformations.

## Contact Information

- **Email**: [raveendranakhila629@gmail.com](mailto:raveendranakhila629@gmail.com)
- **LinkedIn**: [Akhila Raveendran on LinkedIn](https://www.linkedin.com/in/akhila-raveendran)
```
