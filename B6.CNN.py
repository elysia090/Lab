import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loaders(train_csv_path, test_csv_path, batch_size=32):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    X_train = train_df.drop('label', axis=1).values.reshape(-1, 1, 28, 28).astype(float)
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values.reshape(-1, 1, 28, 28).astype(float)
    y_test = test_df['label'].values

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size)

    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.float()  
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss / len(train_loader)))

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float()  
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def test_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer)
    train_accuracy = evaluate_model(model, train_loader)
    test_accuracy = evaluate_model(model, test_loader)
    return train_accuracy, test_accuracy

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders("/kaggle/input/fashionmnist/fashion-mnist_train.csv", "/kaggle/input/fashionmnist/fashion-mnist_test.csv")

    # DI2NNモデルの訓練およびテスト
    input_size = train_loader.dataset[0][0].size(1) * train_loader.dataset[0][0].size(2)
    num_classes = len(set(train_loader.dataset.tensors[1].numpy()))
    di2nn_model = DI2NN(input_size, num_classes)
    di2nn_train_accuracy, di2nn_test_accuracy = test_model(di2nn_model, train_loader, test_loader)

    # CNNモデルの訓練およびテスト
    cnn_model = CNN(num_classes)
    cnn_train_accuracy, cnn_test_accuracy = test_model(cnn_model, train_loader, test_loader)

    print("DI2NN Training Accuracy:", di2nn_train_accuracy)
    print("DI2NN Test Accuracy:", di2nn_test_accuracy)
    print("CNN Training Accuracy:", cnn_train_accuracy)
    print("CNN Test Accuracy:", cnn_test_accuracy)
