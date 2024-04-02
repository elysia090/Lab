import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# BERTモデルとトークナイザーをロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# LSTMモデルの定義
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        # LSTMに入力を与えて隠れ状態を取得
        lstm_out, _ = self.lstm(x)
        # 最後の隠れ状態を返す
        concept_vector = lstm_out[:, -1, :]
        return concept_vector

# LSTMモデルのパラメータ
input_size = 768  # BERTの隠れ状態の次元数
hidden_size = 256  # LSTMの隠れ状態の次元数
num_layers = 2  # LSTMの層数
dropout = 0.2  # ドロップアウト率

# LSTMモデルの初期化
lstm_model = LSTMModel(input_size, hidden_size, num_layers, dropout)

# 概念のテキスト
concept1_text = "I ate apple"
concept2_text = "ornge is good"

# それぞれの概念のテキストをトークン化してBERTに入力
encoded_concept1 = tokenizer(concept1_text, return_tensors='pt', padding=True, truncation=True)
encoded_concept2 = tokenizer(concept2_text, return_tensors='pt', padding=True, truncation=True)

# BERTによるエンコード
with torch.no_grad():
    outputs1 = bert_model(**encoded_concept1)
    outputs2 = bert_model(**encoded_concept2)

# BERTの隠れ状態を取得
hidden_states1 = outputs1.last_hidden_state
hidden_states2 = outputs2.last_hidden_state

# BERTの隠れ状態をLSTMに入力して概念ベクトルを生成
concept_vector1 = lstm_model(hidden_states1)
concept_vector2 = lstm_model(hidden_states2)

# 二つの概念ベクトルの類似度を計算
similarity = torch.nn.functional.cosine_similarity(concept_vector1, concept_vector2, dim=1)

print("類似度:", similarity.item())
