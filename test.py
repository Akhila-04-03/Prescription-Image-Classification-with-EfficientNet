# -*- coding: utf-8 -*-
"""test.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15jZAggt1nH0XAYCnnDBltPQv77OFUF9B
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import models
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from train import PrescriptionDataset  # Import the custom dataset class

# Paths to test directory and CSV file
test_dir = "/content/drive/MyDrive/Testing/testing_words"
test_csv = "/content/drive/MyDrive/Testing/testing_labels.csv"

# Test Transformations
test_transform = Compose([
    Resize(224, 224),
    ToTensorV2(),
])

# Normalization
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Initialize test dataset and data loader
test_dataset = PrescriptionDataset(test_csv, test_dir, transform=test_transform, normalize=normalize)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model with modified classifier for our specific task
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(test_dataset.label_map))

# Load best model weights
model.load_state_dict(torch.load("best_model.pth"))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()  # Set the model to evaluation mode

# Define loss criterion
criterion = torch.nn.CrossEntropyLoss()

# Evaluation function
def evaluate(model, loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# Run evaluation on test data
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")