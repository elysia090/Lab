import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

# JIT 用スクリプト化例（必要に応じて他の頻出関数も対象に）
@torch.jit.script
def select_topk(similarities: torch.Tensor, k_max: int) -> torch.Tensor:
    # similarities: [B, L, num_candidates]
    _, topk_idx = torch.topk(similarities, k=k_max, dim=-1)
    return topk_idx

@dataclass
class FastAttentionConfig:
    d_model: int          # モデル全体の次元
    d_key: int            # キーの次元
    d_query: int          # クエリの次元
    n_heads: int          # ヘッド数
    rank: int             # 低ランク近似のランク
    rff_dim: int          # RFF の出力次元
    k_max: int            # 各クエリごとの最大候補数
    stride: int           # Trie のストライド長
    lsh_buckets: int      # LSH のバケット数
    lsh_bandwidth: float  # LSH のバンド幅
    lsh_key_dim: int      # LSH 入力の次元
    wu_manber_prefix_len: int  # Wu-Manber 用プレフィックス長
    hyper_cuts_dim_groups: Optional[List[int]] = None  # 次元分割用グループ
    n_lsh_hashes: int = 4
    dropout: float = 0.1
    intermediate_dim: int = 2048
    use_rff: bool = True  # RFF を利用するか

# 低ランク線形層 (入力: [B, *, in_features] / 出力: [B, *, out_features])
class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.u_weight = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.v_weight = nn.Parameter(torch.randn(rank, out_features) / math.sqrt(rank))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.u_weight).matmul(self.v_weight)

# Random Fourier Features (入力: [B, *, d_key] / 出力: [B, *, rff_dim])
class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim: int, rff_dim: int):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi, requires_grad=False)
        self.scale = math.sqrt(2.0 / rff_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = x.matmul(self.omega) + self.bias
        return torch.cos(projection) * self.scale

# LSHTable (入力: [B, L, dim] / 出力: [B, L, n_hashes])
class LSHTable(nn.Module):
    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes
        self.random_vectors = nn.Parameter(torch.randn(dim, n_hashes), requires_grad=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.matmul(self.random_vectors)
        return torch.floor(proj / self.bandwidth) % self.n_buckets

_TRIE_INDICES_KEY = '_indices'
# Trie 木 (入力: 単一ベクトル [dim] / 出力: List[int])
class Trie(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.root_node: Dict = {}
        self.stride_len = stride
    def insert(self, binary_vector: torch.Tensor, index: int) -> None:
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())
            if prefix not in current_node:
                current_node[prefix] = {}
            current_node = current_node[prefix]
        if _TRIE_INDICES_KEY not in current_node:
            current_node[_TRIE_INDICES_KEY] = []
        current_node[_TRIE_INDICES_KEY].append(index)
    def search(self, binary_vector: torch.Tensor) -> List[int]:
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())
            if prefix not in current_node:
                return []
            current_node = current_node[prefix]
        return current_node.get(_TRIE_INDICES_KEY, [])

# CandidateFinder: 候補探索（バッチ対応・次元分割対応）
# 入力: query_features, key_features [B, L, group_dim]
# 出力: 候補インデックステンソル [B, L, k_max]
class CandidateFinder(nn.Module):
    def __init__(self, config: FastAttentionConfig, tries: List[Trie], lsh_tables: nn.ModuleList):
        super().__init__()
        self.config = config
        self.tries = tries
        self.lsh_tables = lsh_tables
        self.wu_manber_prefix_len = config.wu_manber_prefix_len
        self.hyper_cuts_dim_groups = config.hyper_cuts_dim_groups

    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        if self.hyper_cuts_dim_groups is None:
            return [features]
        groups = []
        start = 0
        for group_dim in self.hyper_cuts_dim_groups:
            groups.append(features[:, :, start:start+group_dim])
            start += group_dim
        return groups

    def get_lsh_matches_for_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor,
                                  head_idx: int, group_idx: int) -> torch.Tensor:
        # 入力 shape: [B, L, group_dim] → 出力: [B, L, L]
        lsh_table = self.lsh_tables[head_idx][group_idx]
        q_hash = lsh_table(query_grp)  # [B, L, n_hashes]
        k_hash = lsh_table(key_grp)    # [B, L, n_hashes]
        match = (q_hash.unsqueeze(2) == k_hash.unsqueeze(1)).any(dim=-1)
        return match

    def _build_wu_manber_hash_table(self, key_bin: torch.Tensor) -> Dict[tuple, List[int]]:
        # key_bin: [L] の二値ベクトル
        table: Dict[tuple, List[int]] = {}
        L = key_bin.size(0)
        for i in range(L):
            prefix = tuple(key_bin[i, :self.config.wu_manber_prefix_len].tolist())
            table.setdefault(prefix, []).append(i)
        return table

    def _wu_manber_search(self, query_bin: torch.Tensor, table: Dict[tuple, List[int]]) -> List[int]:
        prefix = tuple(query_bin[:self.config.wu_manber_prefix_len].tolist())
        return table.get(prefix, [])

    def _get_wu_manber_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor) -> List[List[List[int]]]:
        B, L, _ = key_grp.size()
        key_bin = self.binary_quantize(key_grp)
        query_bin = self.binary_quantize(query_grp)
        cand_lists = []
        for b in range(B):
            table = self._build_wu_manber_hash_table(key_bin[b])
            batch_list = []
            for i in range(L):
                batch_list.append(self._wu_manber_search(query_bin[b, i], table))
            cand_lists.append(batch_list)
        return cand_lists

    def _get_trie_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> List[List[List[int]]]:
        B, L, _ = key_grp.size()
        cand_lists = []
        for b in range(B):
            trie = Trie(self.config.stride)
            key_bin = self.binary_quantize(key_grp[b])
            for i in range(L):
                trie.insert(key_bin[i], i)
            batch_list = []
            query_bin = self.binary_quantize(query_grp[b])
            for i in range(L):
                batch_list.append(trie.search(query_bin[i]))
            cand_lists.append(batch_list)
        return cand_lists

    def merge_candidate_indices_groups(self, cand_tensors: List[torch.Tensor]) -> torch.Tensor:
        merged = torch.cat(cand_tensors, dim=-1)
        merged, _ = torch.sort(merged)
        return torch.unique_consecutive(merged, dim=-1)

    def _process_dimension_group_candidates(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> torch.Tensor:
        B, L, _ = query_grp.size()
        lsh_match = self.get_lsh_matches_for_group(query_grp, key_grp, head_idx, 0)  # [B, L, L]
        wu_cands = self._get_wu_manber_candidates_group(query_grp, key_grp)
        trie_cands = self._get_trie_candidates_group(query_grp, key_grp, head_idx)
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_grp.device)
        for b in range(B):
            for i in range(L):
                common = list(set(wu_cands[b][i]) & set(trie_cands[b][i]))
                if common:
                    common_tensor = torch.tensor(common, dtype=torch.long, device=query_grp.device)
                    # 候補数が k_max を超える場合は先頭 k_max を採用（後工程で Top-k 選択）
                    if common_tensor.numel() > self.config.k_max:
                        common_tensor = common_tensor[:self.config.k_max]
                    count = common_tensor.numel()
                    candidates[b, i, :count] = common_tensor
        return candidates

    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        # query_up, key_up: [B, L, d] (d は分割後の次元)
        B, L, _ = query_up.size()
        query_groups = self.split_features_by_dim_groups(query_up)
        key_groups = self.split_features_by_dim_groups(key_up)
        cand_list = []
        for grp_idx, (q_grp, k_grp) in enumerate(zip(query_groups, key_groups)):
            cand_grp = self._process_dimension_group_candidates(q_grp, k_grp, head_idx)
            cand_list.append(cand_grp)
        if cand_list:
            merged = self.merge_candidate_indices_groups(cand_list)
            return merged[:, :, :self.config.k_max]
        else:
            return torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_up.device)

# 吸収合成プロジェクション (入力: [B, L, d_query] と [B, L, d_key] → 出力: Q_proj: [B, L, d_key], キーはそのまま)
class AbsorptionProjection(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, rank: int):
        super().__init__()
        self.u_q = nn.Parameter(torch.randn(query_dim, rank) / math.sqrt(rank))
        self.v_q = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))
        self.u_k = nn.Parameter(torch.randn(key_dim, rank) / math.sqrt(rank))
        self.v_k = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        W_UQ = torch.matmul(self.u_q, self.v_q)   # (query_dim, key_dim)
        W_UK = torch.matmul(self.u_k, self.v_k)     # (key_dim, key_dim)
        W_absorb = torch.matmul(W_UK.transpose(0,1), W_UQ)  # (key_dim, key_dim)
        Q_proj = torch.matmul(query, W_absorb.transpose(0,1)) # [B, L, d_key]
        return Q_proj, key  # キーはそのまま

# FastAttention 本体 (入力: query, key, value: [B, L, d_model])
class FastAttention(nn.Module):
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        self.query_down_proj = nn.Linear(config.d_model, config.d_query)
        self.key_value_down_proj = nn.Linear(config.d_model, config.d_key)
        self.absorption_projs = nn.ModuleList([
            AbsorptionProjection(config.d_query, config.d_key, config.rank)
            for _ in range(config.n_heads)
        ])
        self.value_up_projs = nn.ModuleList([
            LowRankLinear(config.d_key, config.d_model, config.rank)
            for _ in range(config.n_heads)
        ])
        self.rff_encoders = nn.ModuleList([
            RandomFourierFeatures(config.d_key, config.rff_dim)
            for _ in range(config.n_heads)
        ])
        self.lsh_tables_list = nn.ModuleList([
            nn.ModuleList([
                LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
                for dim in config.hyper_cuts_dim_groups
            ])
            for _ in range(config.n_heads)
        ]) if config.hyper_cuts_dim_groups else nn.ModuleList([
            nn.ModuleList([
                LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
            ]) for _ in range(config.n_heads)
        ])
        self.tries_list = nn.ModuleList([Trie(config.stride) for _ in range(config.n_heads)])
        self.candidate_finder = CandidateFinder(config, self.tries_list, self.lsh_tables_list)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query, key, value: [B, L, d_model]
        B, L, _ = query.size()
        query_down = self.query_down_proj(query)  # [B, L, d_query]
        key_down = self.key_value_down_proj(key)    # [B, L, d_key]
        head_outputs = []
        for head_idx in range(self.config.n_heads):
            # 吸収合成によるクエリ変換： Q_proj: [B, L, d_key], キーは key_down
            Q_proj, K_proj = self.absorption_projs[head_idx](query_down, key_down)
            # 候補探索
            candidates = self.candidate_finder(Q_proj, K_proj, head_idx)  # [B, L, k_max]
            # ベクトル化した注意計算
            # 安全に gather するため、無効候補 (-1) は 0 に置換
            cand_mask = candidates != -1  # [B, L, k_max]
            safe_candidates = candidates.clone()
            safe_candidates[safe_candidates == -1] = 0
            # 候補キー抽出: gather インデックスは次元1（シーケンス次元）から取得
            candidate_keys = torch.gather(K_proj, 1, safe_candidates.unsqueeze(-1).expand(B, L, self.config.k_max, self.config.d_key))
            # Q_proj の次元拡張: [B, L, 1, d_key]
            q_exp = Q_proj.unsqueeze(2)
            if self.config.use_rff:
                # RFF 変換（適用前に平坦化してから元に戻す）
                q_exp = self.rff_encoders[head_idx](q_exp.view(-1, self.config.d_key)).view(B, L, 1, self.config.rff_dim)
                candidate_keys = self.rff_encoders[head_idx](candidate_keys.view(-1, self.config.d_key)).view(B, L, self.config.k_max, self.config.rff_dim)
                scale = math.sqrt(self.config.rff_dim)
            else:
                scale = math.sqrt(self.config.d_key)
            # 類似度計算: [B, L, k_max]
            sim = torch.matmul(q_exp, candidate_keys.transpose(-2, -1)).squeeze(2) / scale
            # 無効候補は -∞ に
            sim = sim.masked_fill(~cand_mask, float('-inf'))
            attn_weights = F.softmax(sim, dim=-1)  # [B, L, k_max]
            attn_weights = self.dropout(attn_weights)
            # 値抽出: 値側は key_down を元にする（もしくは value を利用する場合も検討可能）
            candidate_values = torch.gather(key_down, 1, safe_candidates.unsqueeze(-1).expand(B, L, self.config.k_max, self.config.d_key))
            # 各ヘッドごとの低ランクアッププロジェクション: 値変換
            candidate_values = self.value_up_projs[head_idx](candidate_values.view(-1, self.config.d_key)).view(B, L, self.config.k_max, self.config.d_model)
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values, dim=2)  # [B, L, d_model]
            head_outputs.append(head_out)
        concat = torch.cat(head_outputs, dim=-1)  # [B, L, d_model * n_heads]
        output = self.output_proj(concat)
        return self.dropout(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.intermediate_dim)
        self.linear2 = nn.Linear(config.intermediate_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class FastAttentionEncoderLayer(nn.Module):
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.self_attn = FastAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = src
        src = self.norm1(src)
        attn = self.self_attn(src, src, src, mask=src_mask)
        src = residual + self.dropout(attn)
        residual = src
        src = self.norm2(src)
        ffn = self.feed_forward(src)
        return residual + self.dropout(ffn)

# 使用例
def example_usage():
    config = FastAttentionConfig(
        d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
        rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
        lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
        hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4, use_rff=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastAttention(config).to(device)
    B, L = 2, 128
    query = torch.randn(B, L, config.d_model).to(device)
    key = torch.randn(B, L, config.d_model).to(device)
    value = torch.randn(B, L, config.d_model).to(device)
    mask = torch.ones(B, L, L, dtype=torch.bool).to(device)
    output = model(query, key, value, mask)
    print(f"FastAttention Output shape: {output.shape}")

def example_usage_encoder_layer():
    config = FastAttentionConfig(
        d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
        rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
        lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
        hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4,
        dropout=0.1, intermediate_dim=2048, use_rff=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_layer = FastAttentionEncoderLayer(config).to(device)
    B, L = 2, 128
    src = torch.randn(B, L, config.d_model).to(device)
    src_mask = torch.ones(B, L, L, dtype=torch.bool).to(device)
    output = encoder_layer(src, src_mask)
    print(f"EncoderLayer Output shape: {output.shape}")

if __name__ == "__main__":
    example_usage()
    example_usage_encoder_layer()
