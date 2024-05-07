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
def load_mnist_dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

    # Reshape input images to the correct size
    train_image = train_dataset.data.float() / 255.0
    test_image = test_dataset.data.float() / 255.0
    train_image = train_image.view(train_image.size(0), -1)
    test_image = test_image.view(test_image.size(0), -1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_image, train_dataset.targets), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_image, test_dataset.targets), batch_size=batch_size, shuffle=False)
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
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, unit_num))
        self.layers.append(nn.Sigmoid())  # Sigmoid activation after binary linear transformation
        for _ in range(layer_num):
            self.layers.append(BinaryLinearED(unit_num, unit_num))
            self.layers.append(nn.Sigmoid())  # Sigmoid activation after binary linear transformation
        self.layers.append(nn.Linear(unit_num, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Model training
def train_and_test_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    model.to(device)
    model.train()
    acc_list = []

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        acc_list.append(train_acc)

    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    test_acc = correct_test / total_test

    return acc_list, train_acc, test_acc

# Main function
def main(layer_num, unit_num, lr, batch_size, epochs):
    train_loader, test_loader = load_mnist_dataset(batch_size)

    # Calculate the input size after flattening the MNIST images
    input_size = 28 * 28

    model = BitNetED(input_size=input_size, output_size=10, layer_num=layer_num, unit_num=unit_num)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    acc_list, train_acc, test_acc = train_and_test_model(model, train_loader, test_loader, criterion, optimizer, epochs)
    end_time = time.time()

    print(f"Finished Training BitNetED in {end_time - start_time:.2f}s")
    print(f"Final Training Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")

    plt.plot(acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('BitNetED Training')
    plt.show()

# Run the main function
if __name__ == "__main__":
    layer_num = 1
    unit_num = 512
    lr = 0.001
    batch_size = 512
    epochs = 20
    main(layer_num, unit_num, lr, batch_size, epochs)
