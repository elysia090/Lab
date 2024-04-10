import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Variational Autoencoder (VAE) の定義
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 784)
        z_params = self.encoder(x)
        mu, logvar = torch.chunk(z_params, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# 状態空間モデルの定義
class StateSpaceModel(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(StateSpaceModel, self).__init__()
        self.transition = nn.Linear(latent_dim, latent_dim)
        self.observation = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        z_next = self.transition(z)
        x_pred = self.observation(z_next)
        return z_next, x_pred

# 学習関数
def train_vae(model, data_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, _ in data_loader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)
        loss = criterion(recon_batch, inputs, mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader.dataset)

# 損失関数の定義
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# データのロード関数
def load_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST(root='./data', train=False, download=True, transform=transform),
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# メイン関数
def main():
    # パラメータ設定
    input_dim = 784
    latent_dim = 20
    epochs = 10
    learning_rate = 1e-3
    batch_size = 64

    # データロード
    train_loader, test_loader = load_data(batch_size=batch_size)

    # モデルの初期化
    vae = VAE(input_dim, latent_dim)
    ssm = StateSpaceModel(latent_dim, input_dim)
    vae_optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    ssm_optimizer = optim.Adam(ssm.parameters(), lr=learning_rate)

    # 学習
    for epoch in range(epochs):
        train_loss = train_vae(vae, train_loader, vae_optimizer, vae_loss)
        print('Epoch: {}, VAE Loss: {:.4f}'.format(epoch+1, train_loss))

    # テスト
    with torch.no_grad():
        for inputs, _ in test_loader:
            recon_batch, mu, logvar = vae(inputs)
            z_next, x_pred = ssm(mu)
            # VAEによって生成された再構成画像を表示
            plt.figure(figsize=(8, 8))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                plt.imshow(recon_batch[i].view(28, 28).cpu().numpy(), cmap='gray')
                plt.axis('off')
            plt.show()
            # 状態空間モデルによって生成された次の状態と観測データを表示
            print("Next state:", z_next[:3])
            print("Predicted observation:", x_pred[:3])

if __name__ == "__main__":
    main()
