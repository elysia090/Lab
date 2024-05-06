import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

# Set seeds for reproducibility
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data preprocessing and loading
def load_cifar10_dataset(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    train_dataset = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10("data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Binary activation function
class BinaryActivation(nn.Module):
    def __init__(self, threshold=0.0):
        super(BinaryActivation, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.sign(x - self.threshold)

# Binary linear layer with ED representation
class BinaryLinearED(nn.Module):
    def __init__(self, in_features, out_features):
        super(BinaryLinearED, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        # Convert weights to binary using ED representation
        binary_weight = torch.sign(self.weight)
        return nn.functional.linear(x, binary_weight, self.bias)

# BitNet model with ED representation and Sigmoid activation
class BitNetED(nn.Module):
    def __init__(self, input_size, output_size, layer_num, unit_num):
        super(BitNetED, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input channels: 3, Output channels: 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling layer
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(32 * 16 * 16, unit_num))  # Adjust input size after convolutions
        self.layers.append(nn.Sigmoid())  # Sigmoid activation after binary linear transformation
        for _ in range(layer_num):
            self.layers.append(BinaryLinearED(unit_num, unit_num))
            self.layers.append(nn.Sigmoid())  # Sigmoid activation after binary linear transformation
        self.layers.append(nn.Linear(unit_num, output_size))

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # Convolution -> ReLU -> Max pooling
        x = x.view(-1, 32 * 16 * 16)  # Reshape for fully connected layer
        for layer in self.layers:
            x = layer(x)
        return x

# Model training
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    model.to(device)
    model.train()

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}")

# Main function
def main(layer_num, unit_num, lr, batch_size, epochs):
    train_loader, test_loader = load_cifar10_dataset(batch_size)

    model = BitNetED(input_size=32 * 16 * 16, output_size=10, layer_num=layer_num, unit_num=unit_num)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    train_model(model, train_loader, test_loader, criterion, optimizer, epochs)
    end_time = time.time()

    print(f"Finished Training BitNetED in {end_time - start_time:.2f}s")

# Run the main function
if __name__ == "__main__":
    layer_num = 1
    unit_num = 64
    lr = 0.001
    batch_size = 128
    epochs = 20
    main(layer_num, unit_num, lr, batch_size, epochs)

