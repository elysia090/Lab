# テキスト生成モデル
import torch
import torch.nn as nn
import torch.optim as optim
import string
import random

# データの準備
all_chars = string.printable
vocab_size = len(all_chars)
char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
idx_to_char = {i: ch for i, ch in enumerate(all_chars)}

# テキスト生成モデルの定義
class TextGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output.view(-1, self.hidden_size))
        return output, hidden

# ランダムなテキスト生成関数の定義
def generate_random_text(model, idx_to_char, char_to_idx, length=100):
    model.eval()
    with torch.no_grad():
        hidden = None
        input_seq = torch.zeros(1, length, dtype=torch.long)
        for i in range(length):
            if i > 0:
                input_seq[0, i] = sampled_char_idx
            output, hidden = model(input_seq[:, i:i+1], hidden)
            output_dist = output.data.view(-1).exp()
            sampled_char_idx = torch.multinomial(output_dist, 1).item()
            if i < length - 1:
                input_seq[0, i+1] = sampled_char_idx
    generated_text = ''.join([idx_to_char[idx] for idx in input_seq.squeeze().tolist()])
    return generated_text

# モデルのトレーニング関数の定義
def train(model, criterion, optimizer, batch_size, num_epochs, seq_length, vocab_size):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        input_seq = torch.randint(0, vocab_size, (seq_length, batch_size)).long()  # 整数型に変換
        target_seq = torch.randint(0, vocab_size, (seq_length * batch_size,)).long()  # 平坦化
        output, _ = model(input_seq, None)
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# パラメータの設定
input_size = vocab_size
hidden_size = 128
output_size = vocab_size
batch_size = 64
num_epochs = 1000
seq_length = 50
learning_rate = 0.001

# モデル、損失関数、最適化アルゴリズムの定義
model = TextGenerator(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# モデルのトレーニング
train(model, criterion, optimizer, batch_size, num_epochs, seq_length, vocab_size)

# テキスト生成
generated_text = generate_random_text(model, idx_to_char, char_to_idx, length=200)
print(generated_text)
