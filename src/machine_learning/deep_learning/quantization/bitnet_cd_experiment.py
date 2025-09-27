import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# Set seeds for reproducibility
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define CustomNet model with regular ReLU activation
class CustomNet(nn.Module):
    def __init__(self, input_channels, output_size, hidden_units):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_size)
        self.relu = nn.ReLU()  # Regular ReLU activation

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))  # Regular ReLU activation for FC1
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Data preprocessing and loading
def load_mnist_dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Set hyperparameters
input_channels = 1  # MNIST has single channel images
output_size = 10  # 10 classes for MNIST digits
hidden_units = 1000  # Number of units in the hidden fully connected layer
batch_size = 512
learning_rate = 0.001
num_epochs = 50

# Load MNIST dataset
train_loader, test_loader = load_mnist_dataset(batch_size)

# Initialize CustomNet model
model = CustomNet(input_channels, output_size, hidden_units).to(device)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store accuracy values
accuracy_values = []
test_accuracy_values = []  

# Training loop
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Calculate test accuracy and store
        test_accuracy = 100 * correct / total
        test_accuracy_values.append(test_accuracy)  # Store test accuracy

    # Calculate training accuracy and store
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    train_accuracy = 100 * correct / total
    accuracy_values.append(train_accuracy)  # Store training accuracy

end_time = time.time()

print(f"Finished Training CustomNet in {end_time - start_time:.2f}s")
print(f"Final Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot accuracy graph
plt.plot(range(1, num_epochs + 1), accuracy_values, marker='o', label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracy_values, marker='s', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.grid(True)
plt.legend()
plt.show()
