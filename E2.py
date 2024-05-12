import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Linear:
    def __init__(self, in_, out_):
        self.weight = torch.rand(in_, out_)
        
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        self.input = x.clone()
        self.output = x @ self.weight
        return self.output

class ED_Layer(torch.nn.Module):
    def __init__(self, in_, out_, alpha=.8):
        super(ED_Layer, self).__init__()
        self.alpha = alpha
        self.pp = torch.nn.Linear(in_, out_)
        self.np = torch.nn.Linear(in_, out_)
        self.pn = torch.nn.Linear(in_, out_)
        self.nn = torch.nn.Linear(in_, out_)
        self.relu = torch.nn.ReLU()
    
    def forward(self, p, n):
        self.op = self.relu(self.pp(p) - self.np(n))
        self.on = self.relu(self.nn(n) - self.pn(p))
        return self.op, self.on
    
    def update(self, d):
        for ope, l, o in ((1, self.pp, self.op), (-1, self.np, self.op), (1, self.pn, self.on), (-1, self.nn, self.on)):
            dw = torch.einsum("b,bo,bi->bio", d.mean(1), torch.where(o > 0, torch.tensor(1.0), torch.tensor(0.0)), l.input)
            l.weight += ope * self.alpha * dw.mean(0)
                
class ED_OutputLayer(ED_Layer):
    def __init__(self, in_, out_, alpha=.8):
        super(ED_OutputLayer, self).__init__(in_, out_, alpha)
        
    def update(self, d):
        for ope, l, o in ((1, self.pp, self.op), (-1, self.np, self.op), (1, self.pn, self.on), (-1, self.nn, self.on)):
            dw = torch.einsum("bo,bo,bi->bio", torch.where(d > 0, torch.tensor(1.0), torch.tensor(0.0)), torch.where(o > 0, torch.tensor(1.0), torch.tensor(0.0)), l.input)
            l.weight += ope * self.alpha * dw.mean(0)
            
        
class ED(torch.nn.Module):
    def __init__(self, in_, out_, hidden_width, hidden_depth=1, alpha=.8):
        super(ED, self).__init__()
        self.layers = torch.nn.ModuleList([
            ED_Layer(in_, hidden_width, alpha),
            *[ED_Layer(hidden_width, hidden_width, alpha) for _ in range(hidden_depth)],
            ED_OutputLayer(hidden_width, out_, alpha)
        ])
    
    def forward(self, x):
        p, n = x[0].clone(), x[1].clone()
        for layer in self.layers:
            p, n = layer(p, n)
        return F.log_softmax(p, dim=1)  # Apply log softmax
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def train(self, mode=True):
        super(ED, self).train(mode)
        for layer in self.layers:
            layer.train(mode)
    
    def eval(self):
        super(ED, self).eval()
        for layer in self.layers:
            layer.eval()

class MNISTLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_loader, self.test_loader = self.load_mnist_dataset()

    def load_mnist_dataset(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

# Lists to store accuracy values
accuracy_values = []
test_accuracy_values = []  

# Training loop
def train(model, train_loader, test_loader, num_epochs, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.NLLLoss()
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training
        for inputs, targets in train_loader:
            inputs = inputs.to(device)  # Move inputs to the appropriate device
            targets = targets.to(device)  # Move targets to the appropriate device
            
            optimizer.zero_grad()
            positive_inputs = inputs.clone().reshape(inputs.shape[0], -1)
            negative_inputs = inputs.clone().reshape(inputs.shape[0], -1)
            outputs = model((positive_inputs, negative_inputs))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_accuracy = correct / total
        accuracy_values.append(train_accuracy)
        
        # Evaluation on test set
        model.eval()  # Set the model to evaluation mode
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)  # Move inputs to the appropriate device
                targets = targets.to(device)  # Move targets to the appropriate device
                
                positive_inputs = inputs.clone().reshape(inputs.shape[0], -1)
                negative_inputs = inputs.clone().reshape(inputs.shape[0], -1)
                outputs = model((positive_inputs, negative_inputs))
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_accuracy = test_correct / test_total
        test_accuracy_values.append(test_accuracy)
        
        # Print training statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {running_loss / len(train_loader):.4f}, '
              f'Training Accuracy: {train_accuracy:.4f}, '
              f'Test Accuracy: {test_accuracy:.4f}')

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), accuracy_values, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracy_values, label='Test Accuracy')
    plt.title('Training and Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Example usage:
batch_size = 64
num_epochs = 20
learning_rate = 0.001

train_loader, test_loader = MNISTLoader(batch_size).train_loader, MNISTLoader(batch_size).test_loader
model = ED(in_=28*28, out_=10, hidden_width=1000, hidden_depth=1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train(model, train_loader, test_loader, num_epochs, optimizer)
