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

# Define CustomNet model with custom ReLU activation
class CustomNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Define custom weight matrices based on the concept
        self.forward_ope_fc1 = torch.tensor([[1, -1], [-1, 1]])  # Example for the first layer
        self.forward_ope_fc2 = torch.tensor([[1, -1], [-1, 1]])  # Example for the second layer

        # Initialize custom weight matrices
        self.update_p_ope_fc1 = torch.tensor([[1, 0], [0, 0]])  # Example for the first layer
        self.update_n_ope_fc1 = torch.tensor([[0, 0], [0, 1]])  # Example for the first layer
        self.update_p_ope_fc2 = torch.tensor([[1, 0], [0, 0]])  # Example for the second layer
        self.update_n_ope_fc2 = torch.tensor([[0, 0], [0, 1]])  # Example for the second layer

    def custom_relu(self, x):
        return (torch.sign(x) + F.relu(x)) * (1 - torch.exp(-torch.abs(x)))

    def forward(self, x):
        x = self.custom_relu(self.fc1(x))  # Custom ReLU activation function
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def update_weights(self, loss, lr):
        # Update weights of the first layer
        w_fc1 = (self.forward_ope_fc1 * loss * self.recent_in.grad * self.recent_out.grad).sum(dim=0)
        self.fc1.weight += lr * (self.update_p_ope_fc1 * w_fc1 / self.fc1.out_features)
        # Update weights of the second layer
        w_fc2 = (self.forward_ope_fc2 * loss * self.fc1.recent_out.grad * self.recent_out.grad).sum(dim=0)
        self.fc2.weight += lr * (self.update_p_ope_fc2 * w_fc2 / self.fc2.out_features)

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
hidden_size = 1024
output_size = 10  # 10 classes for MNIST digits
batch_size = 512
learning_rate = 0.001
num_epochs = 50

# Load MNIST dataset
train_loader, test_loader = load_mnist_dataset(batch_size)

# Initialize CustomNet model
model = CustomNet(input_size, hidden_size, output_size).to(device)

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

        # Store gradients for weight updates
        for name, param in model.named_parameters():
            if name == 'fc1.weight':
                model.fc1.recent_in = data
                model.fc1.recent_out = output
            elif name == 'fc2.weight':
                model.fc2.recent_in = model.fc1.recent_out
                model.fc2.recent_out = output

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
