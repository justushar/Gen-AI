import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_splitz
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision.models import mobilenet_v2

# Define paths to the training, validation, and test datasets
train_path = "/kaggle/input/papsinglecell/SingleCellPAP/Training"
test_path = "/kaggle/input/papsinglecell/SingleCellPAP/Test"

# Function to load and preprocess images
def load_images_and_labels(path):
    images = []
    labels = []
    class_names = os.listdir(path)
    for class_name in class_names:
        class_path = os.path.join(path, class_name)
        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (128, 128))  # Resize to ResNet50 input size
            images.append(img)
            labels.append(class_name)
    return np.array(images), np.array(labels)

# Load images and labels for training and test sets
x_train, y_train = load_images_and_labels(train_path)
x_test, y_test = load_images_and_labels(test_path)

# Optionally, encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train_encoded))