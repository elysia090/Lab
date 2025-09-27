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

# Define BitNet model with sigmoid activation
class BitNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BitNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Sigmoid activation function
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

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

# Set hyperparameters
input_size = 28 * 28  # MNIST image size
hidden_size = 512
output_size = 10  # 10 classes for MNIST digits
batch_size = 512
learning_rate = 0.001
num_epochs = 50

# Load MNIST dataset
train_loader, test_loader = load_mnist_dataset(batch_size)

# Initialize BitNet model
model = BitNet(input_size, hidden_size, output_size).to(device)

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

print(f"Finished Training BitNet in {end_time - start_time:.2f}s")
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
