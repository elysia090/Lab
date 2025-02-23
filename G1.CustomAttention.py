import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

@dataclass
class FastAttentionConfig:
    """FastAttentionの設定 (hyper-Cuts 次元分割対応)"""
    d_model: int
    d_key: int
    d_query: int
    n_heads: int
    rank: int
    rff_dim: int
    k_max: int
    stride: int
    lsh_buckets: int
    lsh_bandwidth: float
    lsh_key_dim: int
    wu_manber_prefix_len: int
    hyper_cuts_dim_groups: Optional[List[int]] = None
    n_lsh_hashes: int = 4
    dropout: float = 0.1
    intermediate_dim: int = 2048

class LowRankLinear(nn.Module):
    """低ランク線形層"""
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.u_weight = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.v_weight = nn.Parameter(torch.randn(rank, out_features) / math.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.u_weight).matmul(self.v_weight)

class RandomFourierFeatures(nn.Module):
    """ランダムフーリエ特徴"""
    def __init__(self, input_dim: int, rff_dim: int):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi, requires_grad=False)
        self.scale = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = x.matmul(self.omega) + self.bias
        return torch.cos(projection) * self.scale

class LSHTable(nn.Module):
    """LSHテーブル (複数ハッシュ関数対応)"""
    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes
        self.random_vectors = nn.Parameter(torch.randn(dim, n_hashes), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hash(x)

    def hash(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.matmul(self.random_vectors)
        return torch.floor(proj / self.bandwidth) % self.n_buckets

_TRIE_INDICES_KEY = '_indices'

class Trie(nn.Module):
    """Trie木"""
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

class CandidateFinder(nn.Module):
    """候補探索器 (Wu-Manber + Trie木 + hyper-Cuts 次元分割)"""
    def __init__(self, config: FastAttentionConfig, tries: List[Trie], lsh_tables: nn.ModuleList):
        super().__init__()
        self.config = config
        self.tries = tries
        self.lsh_tables = lsh_tables
        self.wu_manber_prefix_len = config.wu_manber_prefix_len
        self.hyper_cuts_dim_groups = config.hyper_cuts_dim_groups

    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """特徴量を二値量子化"""
        return (x > 0).float()

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        """特徴量を次元グループごとに分割 (hyper-Cuts用)"""
        if self.hyper_cuts_dim_groups is None:
            return [features]

        dim_groups = []
        start_dim = 0
        for group_dim in self.hyper_cuts_dim_groups:
            end_dim = start_dim + group_dim
            dim_groups.append(features[:, :, start_dim:end_dim])
            start_dim = end_dim
        return dim_groups

    def get_lsh_matches_for_group(self, query_features_group: torch.Tensor, key_features_group: torch.Tensor, head_idx: int, group_idx: int) -> torch.Tensor:
        """次元グループごとのLSHマッチング (複数ハッシュ関数対応)"""
        lsh_table_for_group = self.lsh_tables[head_idx][group_idx]
        query_hashes = lsh_table_for_group(query_features_group) # [batch_size, seq_len, n_lsh_hashes]
        key_hashes = lsh_table_for_group(key_features_group)     # [batch_size, seq_len, n_lsh_hashes]
        match_matrix = (query_hashes.unsqueeze(2) == key_hashes.unsqueeze(1)).any(dim=-1) # [batch_size, seq_len, seq_len]
        return match_matrix

    def _build_wu_manber_hash_table(self, key_binary_batch: torch.Tensor, prefix_len: int) -> Dict[tuple, List[int]]:
        """Wu-Manber ハッシュテーブルを構築 (内部関数)"""
        hash_table: Dict[tuple, List[int]] = {}
        seq_len = key_binary_batch.size(1)
        for key_index in range(seq_len):
            key_vec = key_binary_batch[0, key_index]
            prefix = tuple(key_vec[:prefix_len].tolist())
            if prefix not in hash_table:
                hash_table[prefix] = []
            hash_table[prefix].append(key_index)
        return hash_table

    def _wu_manber_search(self, query_binary_vec: torch.Tensor, wu_manber_hash_table: Dict[tuple, List[int]], prefix_len: int) -> List[int]:
        """Wu-Manber アルゴリズムで候補検索 (内部関数)"""
        prefix = tuple(query_binary_vec[:prefix_len].tolist())
        return wu_manber_hash_table.get(prefix, [])

    def _get_wu_manber_candidates_group(self, query_features_group: torch.Tensor, key_features_group: torch.Tensor, head_idx: int) -> List[List[int]]:
        """次元グループごとのWu-Manber初期候補取得 (内部関数)"""
        key_binary_batch = self.binary_quantize(key_features_group)
        wu_manber_hash_table = self._build_wu_manber_hash_table(key_binary_batch, self.config.wu_manber_prefix_len)
        seq_len = query_features_group.size(1)
        initial_candidates_list: List[List[int]] = []
        query_binary_batch = self.binary_quantize(query_features_group)
        for query_index in range(seq_len):
            query_vec_binary = query_binary_batch[0, query_index]
            initial_candidates_list.append(self._wu_manber_search(query_vec_binary, wu_manber_hash_table, self.config.wu_manber_prefix_len))
        return initial_candidates_list

    def _get_trie_candidates_group(self, batch_idx: int, query_features_group: torch.Tensor, key_features_group: torch.Tensor, head_idx: int, tries_built_for_batch: Dict[int, bool]) -> List[List[int]]:
        """次元グループごとのTrie木候補絞り込み (内部関数)"""
        trie = self.tries[head_idx]

        if head_idx not in tries_built_for_batch:
            trie.root_node = {} # Trie reset per head and batch
            key_binary_batch = self.binary_quantize(key_features_group)
            seq_len = key_features_group.size(1)
            for key_index in range(seq_len):
                trie.insert(key_binary_batch[batch_idx, key_index], key_index)
            tries_built_for_batch[head_idx] = True

        trie_candidates_list: List[List[int]] = []
        query_binary_batch = self.binary_quantize(query_features_group)
        seq_len = query_features_group.size(1)
        for query_index in range(seq_len):
            query_vec_binary = query_binary_batch[batch_idx, query_index]
            trie_candidates = trie.search(query_vec_binary)
            trie_candidates_list.append(trie_candidates)
        return trie_candidates_list

    def merge_candidate_indices_groups(self, candidate_indices_groups: List[torch.Tensor]) -> torch.Tensor:
        """次元グループごとの候補インデックスを統合 (重複削除)"""
        if not candidate_indices_groups:
            device = candidate_indices_groups[0].device if candidate_indices_groups else torch.device('cpu') # 空リスト対策でdevice取得
            return torch.empty((candidate_indices_groups[0].size(0), candidate_indices_groups[0].size(1), 0), dtype=torch.long, device=device)

        merged_candidates = torch.cat(candidate_indices_groups, dim=-1)
        merged_candidates, _ = torch.sort(merged_candidates)
        return torch.unique_consecutive(merged_candidates, dim=-1)

    def select_top_k_candidates_for_query(self, combined_candidate_indices: torch.Tensor, query_features_group: torch.Tensor, key_features_group: torch.Tensor, batch_idx: int, query_idx: int) -> torch.Tensor:
        """Top-k 候補を選択 (クエリごと)"""
        k_max = self.config.k_max
        if len(combined_candidate_indices) <= k_max:
            return combined_candidate_indices

        similarities = torch.matmul(key_features_group[batch_idx, combined_candidate_indices], query_features_group[batch_idx, query_idx].unsqueeze(0).transpose(0, 1)).squeeze(1)
        _, top_indices = torch.topk(similarities, k_max)
        return combined_candidate_indices[top_indices]

    def _process_dimension_group_candidates(self, query_features_group: torch.Tensor, key_features_group: torch.Tensor, head_idx: int, group_idx: int, batch_size: int, seq_len: int, tries_built_for_batch: Dict[int, bool]) -> torch.Tensor:
        """次元グループに対する候補探索処理 (内部関数): LSH, Wu-Manber, Trie木を統合"""
        lsh_match_matrix = self.get_lsh_matches_for_group(query_features_group, key_features_group, head_idx, group_idx) # LSH マッチング
        wu_manber_initial_candidates_lists = self._get_wu_manber_candidates_group(query_features_group, key_features_group, head_idx) # Wu-Manber 初期候補
        trie_candidate_lists = self._get_trie_candidates_group(batch_size - 1, query_features_group, key_features_group, head_idx, tries_built_for_batch) # Trie木で候補絞り込み
        # trie_candidate_lists = self._get_trie_candidates_group(batch_size - 1, query_features_group, key_features_group, head_idx, wu_manber_initial_candidates_lists, tries_built_for_batch) # TypeError発生箇所 (修正前)

        candidates_group = torch.full((batch_size, seq_len, self.config.k_max), -1, dtype=torch.long, device=query_features_group.device)

        for batch_idx in range(batch_size):
            for query_idx in range(seq_len):
                matched_indices_tensor = lsh_match_matrix[batch_idx, query_idx].nonzero(as_tuple=False).squeeze(1)
                if matched_indices_tensor.numel() > 0:
                    trie_candidates = trie_candidate_lists[query_idx]
                    trie_candidates_tensor = torch.tensor(trie_candidates, dtype=torch.long, device=query_features_group.device)
                    combined_candidate_indices = trie_candidates_tensor[torch.isin(trie_candidates_tensor, matched_indices_tensor)] # LSHとTrieの共通候補
                    if combined_candidate_indices.numel() > 0:
                        top_k_indices = self.select_top_k_candidates_for_query(combined_candidate_indices, query_features_group, key_features_group, batch_idx, query_idx) # Top-k 選択
                        candidate_count = min(len(top_k_indices), self.config.k_max)
                        candidates_group[batch_idx, query_idx, :candidate_count] = top_k_indices[:candidate_count]
        return candidates_group


    def forward(self, query_features_up: torch.Tensor, key_features_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        """候補探索器のForward処理"""
        batch_size, seq_len, _ = query_features_up.size()
        query_features_groups = self.split_features_by_dim_groups(query_features_up)
        key_features_groups = self.split_features_by_dim_groups(key_features_up)

        candidate_indices_groups: List[torch.Tensor] = []
        tries_built_for_batch: Dict[int, bool] = {} # バッチ内で Trie が構築されたかを追跡

        for group_idx, (query_features_group, key_features_group) in enumerate(zip(query_features_groups, key_features_groups)):
            candidates_group = self._process_dimension_group_candidates(query_features_group, key_features_group, head_idx, group_idx, batch_size, seq_len, tries_built_for_batch)
            candidate_indices_groups.append(candidates_group)

        if not candidate_indices_groups: # 次元分割がない場合などの空リスト対策
            return torch.full((batch_size, seq_len, self.config.k_max), -1, dtype=torch.long, device=query_features_up.device)

        candidates = self.merge_candidate_indices_groups(candidate_indices_groups)
        return candidates[:, :, :self.config.k_max]


class FastAttention(nn.Module):
    """高速注意機構 (Wu-Manber + Trie木 + hyper-Cuts 次元分割)"""
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        self.query_down_proj = nn.Linear(config.d_model, config.d_query)
        self.key_value_down_proj = nn.Linear(config.d_model, config.d_key)
        self.query_up_projs = nn.ModuleList([LowRankLinear(config.d_query, config.d_key, config.rank) for _ in range(config.n_heads)])
        self.key_up_projs = nn.ModuleList([LowRankLinear(config.d_key, config.d_key, config.rank) for _ in range(config.n_heads)])
        self.value_up_projs = nn.ModuleList([LowRankLinear(config.d_key, config.d_model, config.rank) for _ in range(config.n_heads)])
        self.rff_encoders = nn.ModuleList([RandomFourierFeatures(config.d_key, config.rff_dim) for _ in range(config.n_heads)])

        self.lsh_tables_list = nn.ModuleList([
            nn.ModuleList([
                LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
                for dim in config.hyper_cuts_dim_groups
            ])
            for _ in range(config.n_heads)
        ]) if config.hyper_cuts_dim_groups else nn.ModuleList([
            nn.ModuleList([LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)]) for _ in range(config.n_heads)
        ])

        self.tries_list = nn.ModuleList([Trie(config.stride) for _ in range(config.n_heads)])
        self.candidate_finder = CandidateFinder(config, self.tries_list, self.lsh_tables_list)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        query_down = self.query_down_proj(query)
        key_value_down = self.key_value_down_proj(key)
        head_outputs: List[torch.Tensor] = []

        for head_idx in range(self.config.n_heads):
            query_features_up = self.query_up_projs[head_idx](query_down)
            key_features_up = self.key_up_projs[head_idx](key_value_down)
            candidates = self.candidate_finder(query_features_up, key_features_up, head_idx)

            head_output = torch.zeros(batch_size, seq_len, self.config.d_model, device=query.device)
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    valid_candidate_indices = candidates[batch_idx, seq_idx][candidates[batch_idx, seq_idx] != -1]
                    if valid_candidate_indices.numel() > 0:
                        q_up = query_features_up[batch_idx, seq_idx].unsqueeze(0)
                        k_up = key_features_up[batch_idx, valid_candidate_indices]
                        scores = q_up.matmul(k_up.transpose(-2, -1)) / math.sqrt(self.config.d_key)

                        if mask is not None:
                            mask_slice = mask[batch_idx, seq_idx, :seq_len]
                            scores = scores.masked_fill(~mask_slice[valid_candidate_indices].unsqueeze(0), float('-inf'))

                        attn_weights = F.softmax(scores, dim=-1)
                        attn_weights = self.dropout(attn_weights)
                        v_up = self.value_up_projs[head_idx](key_value_down[batch_idx, valid_candidate_indices])
                        head_output[batch_idx, seq_idx] = attn_weights.matmul(v_up).squeeze(0)
            head_outputs.append(head_output)

        concat_output = torch.cat(head_outputs, dim=-1)
        output = self.output_proj(concat_output)
        return self.dropout(output)


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network (EncoderLayer連携のため追加)"""
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
    """FastAttention EncoderLayer (EncoderLayer連携のため追加)"""
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
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = residual + self.dropout(attn_output)

        residual = src
        src = self.norm2(src)
        ffn_output = self.feed_forward(src)
        src = residual + self.dropout(ffn_output)
        return src


def example_usage():
    config = FastAttentionConfig(d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
                                  rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
                                  lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
                                  hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastAttention(config).to(device)
    batch_size, seq_len = 2, 128
    query = torch.randn(batch_size, seq_len, config.d_model).to(device)
    key = torch.randn(batch_size, seq_len, config.d_model).to(device)
    value = torch.randn(batch_size, seq_len, config.d_model).to(device)
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool).to(device)
    output = model(query, key, value, mask)
    print(f"FastAttention Output shape: {output.shape}")


def example_usage_encoder_layer():
    config = FastAttentionConfig(d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
                                  rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
                                  lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
                                  hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4,
                                  dropout=0.1, intermediate_dim=2048)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_layer = FastAttentionEncoderLayer(config).to(device)
    batch_size, seq_len = 2, 128
    src = torch.randn(batch_size, seq_len, config.d_model).to(device)
    src_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool).to(device)
    output = encoder_layer(src, src_mask)
    print(f"EncoderLayer Output shape: {output.shape}")


if __name__ == "__main__":
    example_usage()
    example_usage_encoder_layer()
