import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deap import base, creator, tools, algorithms
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Reduce dataset size
small_dataset_size = 1000  # Use only 1000 samples
indices = list(range(len(full_dataset)))
random.shuffle(indices)
small_indices = indices[:small_dataset_size]

small_dataset = torch.utils.data.Subset(full_dataset, small_indices)

# Create DataLoader
train_loader = torch.utils.data.DataLoader(small_dataset, batch_size=32, shuffle=True)
