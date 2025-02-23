import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 低ランク近似のためのクラス
class LowRankMatrix(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankMatrix, self).__init__()
        self.U = nn.Parameter(torch.randn(out_features, rank))  # (out_features, rank)
        self.V = nn.Parameter(torch.randn(rank, in_features))  # (rank, in_features)
    
    def forward(self, x):
        # x: (T, in_features) → 输出: (T, out_features)
        return torch.matmul(self.U, torch.matmul(self.V, x.t())).t()

# ランダムフーリエ特徴 (RFF) のクラス
class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandomFourierFeatures, self).__init__()
        # ランダムな射影行列とバイアス
        self.omega = nn.Parameter(torch.randn(input_dim, output_dim) * (2 * np.pi))
        self.b = nn.Parameter(torch.rand(output_dim) * 2 * np.pi)
    
    def forward(self, x):
        # x: (T, input_dim) → 输出: (T, output_dim)
        return np.sqrt(2 / self.omega.shape[1]) * torch.cos(torch.matmul(x, self.omega) + self.b)

# LSH (Locality-Sensitive Hashing) のクラス
class LSH:
    def __init__(self, input_dim, num_buckets, w):
        self.num_buckets = num_buckets
        self.w = w
        self.r = torch.randn(input_dim)  # ランダムな射影ベクトル
    
    def hash(self, x):
        # x: (input_dim,) → ハッシュ値 (整数)
        return int((torch.sum(x * self.r) / self.w).item()) % self.num_buckets

# 統合モデル
class IntegratedModel(nn.Module):
    def __init__(self, d, d_c, d_c_prime, d_h, r, num_heads, k_max, num_buckets, w):
        super(IntegratedModel, self).__init__()
        # ハイパーパラメータの設定
        self.d = d              # 入力次元
        self.d_c = d_c          # KV側の圧縮次元
        self.d_c_prime = d_c_prime  # クエリ側の圧縮次元
        self.d_h = d_h          # ヘッドごとの出力次元
        self.r = r              # 低ランク近似のランク
        self.num_heads = num_heads  # ヘッド数
        self.k_max = k_max      # 候補集合の最大サイズ
        self.num_buckets = num_buckets  # LSHのバケット数
        self.w = w              # LSHの量子化幅
        
        # ダウンプロジェクション用の線形層
        self.W_DQ = nn.Linear(d, d_c_prime)  # クエリ用
        self.W_DKV = nn.Linear(d, d_c)       # キー・バリュー用
        
        # 各ヘッドごとのアッププロジェクション（低ランク近似）
        self.UQ = nn.ModuleList([LowRankMatrix(d_c_prime, d_h, r) for _ in range(num_heads)])
        self.UK = nn.ModuleList([LowRankMatrix(d_c, d_h, r) for _ in range(num_heads)])
        self.UV = nn.ModuleList([LowRankMatrix(d_c, d_h, r) for _ in range(num_heads)])
        self.WO = nn.ModuleList([LowRankMatrix(d_h, d, r) for _ in range(num_heads)])
        
        # ランダムフーリエ特徴
        self.phi = RandomFourierFeatures(d_c, r)
        
        # LSH
        self.lsh = LSH(d_c, num_buckets, w)
        
        # 最終出力用の線形層
        self.W_final = nn.Linear(num_heads * d, d)
    
    def forward(self, X):
        T = X.shape[0]  # シーケンス長
        
        # 1. ダウンプロジェクション
        C_Q = self.W_DQ(X)    # (T, d_c_prime)
        C_KV = self.W_DKV(X)  # (T, d_c)
        
        # マルチヘッド出力のリスト
        O = []
        for i in range(self.num_heads):
            # 2. アッププロジェクションと吸収
            Q_prime = self.UQ[i](C_Q)  # (T, d_h)
            K_prime = self.UK[i](C_KV) # (T, d_h)
            
            # 3. 高速注意機構 (RFF)
            tilde_Q = self.phi(C_KV)   # (T, r)
            tilde_K = self.phi(C_KV)   # (T, r)
            
            # 4. 離散化とLSHによる候補抽出
            bar_Q = torch.sign(Q_prime)  # (T, d_h)
            bar_K = torch.sign(K_prime)  # (T, d_h)
            
            # キーのハッシュ値を計算
            h_k = [self.lsh.hash(bar_K[k]) for k in range(T)]
            
            # クエリごとの候補集合を抽出
            I_j = []
            for j in range(T):
                h_q = self.lsh.hash(bar_Q[j])
                candidates = [k for k in range(T) if h_k[k] == h_q]
                if len(candidates) > self.k_max:
                    # 候補が多い場合はスコアで制限
                    scores = torch.matmul(tilde_Q[j], tilde_K[candidates].t())
                    top_k_indices = torch.topk(scores, self.k_max).indices
                    candidates = [candidates[idx] for idx in top_k_indices]
                I_j.append(candidates)
            
            # 5. 候補制限付き注意計算
            L_ij = []
            for j in range(T):
                q_j = Q_prime[j]                 # (d_h)
                k_candidates = K_prime[I_j[j]]   # (k_max, d_h)
                scores = torch.matmul(q_j, k_candidates.t()) / np.sqrt(self.d_h)
                L_ij.append(scores)
            
            A_ij = [F.softmax(l, dim=0) for l in L_ij]  # 注意重み
            
            # バリュー計算
            V_i = self.UV[i](C_KV)  # (T, d_h)
            O_i = []
            for j in range(T):
                weighted_sum = torch.sum(A_ij[j].unsqueeze(1) * V_i[I_j[j]], dim=0)
                O_i.append(weighted_sum)
            O_i = torch.stack(O_i)  # (T, d_h)
            
            # 出力プロジェクション
            O_i = self.WO[i](O_i)   # (T, d)
            O.append(O_i)
        
        # 6. 全ヘッドの出力を結合
        O = torch.cat(O, dim=1)  # (T, num_heads * d)
        
        # 7. 最終出力
        O = self.W_final(O)      # (T, d)
        
        return O

# 使用例
d = 512           # 入力次元
d_c = 256         # KV圧縮次元
d_c_prime = 128   # クエリ圧縮次元
d_h = 64          # ヘッド出力次元
r = 32            # 低ランク近似のランク
num_heads = 8     # ヘッド数
k_max = 10        # 最大候補数
num_buckets = 100 # LSHバケット数
w = 1.0           # LSH量子化幅

# モデルのインスタンス化
model = IntegratedModel(d, d_c, d_c_prime, d_h, r, num_heads, k_max, num_buckets, w)
X = torch.randn(100, d)  # 入力例: シーケンス長100
output = model(X)
print(output.shape)  # 出力形状: (100, 512)
