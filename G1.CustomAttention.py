import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
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
    n_lsh_hashes: int = 4 # LSHハッシュ関数の数 (追加)

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
    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int): # n_hashes を追加
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes # ハッシュ関数の数
        self.random_vectors = nn.Parameter(torch.randn(dim, n_hashes), requires_grad=False) # 複数のランダムベクトル

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hash(x)

    def hash(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.matmul(self.random_vectors) # 複数のランダムベクトルで射影
        return torch.floor(proj / self.bandwidth) % self.n_buckets # 各ハッシュ関数でハッシュ値を計算

_TRIE_INDICES_KEY = '_indices' # Trieノードのインデックスリストのキーを定数として定義

class Trie(nn.Module):
    """Trie木"""
    def __init__(self, stride: int):
        super().__init__()
        self.root_node = {}
        self.stride_len = stride

    def insert(self, binary_vector: torch.Tensor, index: int):
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
        return (x > 0).float()

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        """特徴量を次元グループごとに分割"""
        if self.hyper_cuts_dim_groups is None:
            return [features]

        dim_groups = []
        start_dim = 0
        for group_dim in self.hyper_cuts_dim_groups:
            end_dim = start_dim + group_dim
            dim_groups.append(features[:, :, start_dim:end_dim])
            start_dim = end_dim
        return dim_groups

    def get_lsh_matches(self, query_up_group: torch.Tensor, key_up_group: torch.Tensor, head_idx: int, group_idx: int) -> torch.Tensor:
        """次元グループごとのLSHマッチング (複数ハッシュ関数対応)"""
        lsh_table_for_group = self.lsh_tables[head_idx][group_idx]
        query_hashes = lsh_table_for_group(query_up_group) # [batch_size, seq_len, n_lsh_hashes]
        key_hashes = lsh_table_for_group(key_up_group)     # [batch_size, seq_len, n_lsh_hashes]

        # 各ハッシュ関数で一致するかどうかをチェックし、いずれかのハッシュ関数で一致すればマッチとみなす (OR条件)
        matches = (query_hashes.unsqueeze(2) == key_hashes.unsqueeze(1)).any(dim=-1) # [batch_size, seq_len, seq_len]
        return matches


    def build_wu_manber_hash_table(self, key_binary_batch: torch.Tensor, prefix_len: int) -> dict:
        """Wu-Manber ハッシュテーブルを構築"""
        hash_table = {}
        seq_len = key_binary_batch.size(1)
        for key_index in range(seq_len):
            key_vec = key_binary_batch[0, key_index] # GPU上でのテンソルを維持
            prefix = tuple(key_vec[:prefix_len].tolist()) # CPUへの転送はprefixのみ
            if prefix not in hash_table:
                hash_table[prefix] = []
            hash_table[prefix].append(key_index)
        return hash_table

    def wu_manber_search(self, query_binary_vec: torch.Tensor, wu_manber_hash_table: dict, prefix_len: int) -> List[int]:
        """Wu-Manber アルゴリズムで候補検索"""
        prefix = tuple(query_binary_vec[:prefix_len].tolist())
        return wu_manber_hash_table.get(prefix, [])

    def _get_initial_candidates_wu_manber_group(self, query_up_group: torch.Tensor, key_up_group: torch.Tensor, head_idx: int) -> List[List[int]]:
        """次元グループごとのWu-Manber初期候補取得 (内部関数)"""
        key_binary_batch = self.binary_quantize(key_up_group)
        wu_manber_hash_table = self.build_wu_manber_hash_table(key_binary_batch, self.config.wu_manber_prefix_len)
        seq_len = query_up_group.size(1)
        initial_candidates_list = []
        query_binary_batch = self.binary_quantize(query_up_group) # Queryもここで量子化
        for query_index in range(seq_len):
            query_vec_binary = query_binary_batch[0, query_index] # GPU上でのテンソルを維持
            initial_candidates_list.append(self.wu_manber_search(query_vec_binary, wu_manber_hash_table, self.config.wu_manber_prefix_len))
        return initial_candidates_list

    def _get_trie_candidates_group(self, batch_idx: int, query_up_group: torch.Tensor, key_up_group: torch.Tensor, head_idx: int, initial_candidates_list: List[List[int]], tries_built_for_batch: dict) -> List[List[int]]:
        """次元グループごとのTrie木候補絞り込み (内部関数)"""
        trie = self.tries[head_idx]

        # バッチ内で Trie がまだ構築されていなければ構築
        if head_idx not in tries_built_for_batch:
            trie.root_node = {} # Trie reset per head and batch
            key_binary_batch = self.binary_quantize(key_up_group)
            seq_len = key_up_group.size(1)
            for key_index in range(seq_len):
                trie.insert(key_binary_batch[batch_idx, key_index], key_index) # GPU上Tensorを直接渡す
            tries_built_for_batch[head_idx] = True # head_idx を辞書に追加して構築済みを記録

        trie_candidates_list = []
        query_binary_batch = self.binary_quantize(query_up_group) # Queryもここで量子化
        seq_len = query_up_group.size(1)
        for query_index in range(seq_len):
            query_vec_binary = query_binary_batch[batch_idx, query_index] # GPU上Tensorを直接渡す
            trie_candidates = trie.search(query_vec_binary)
            trie_candidates_list.append(trie_candidates)
        return trie_candidates_list

    def merge_candidate_indices(self, candidate_indices_groups: List[torch.Tensor]) -> torch.Tensor:
        """次元グループごとの候補インデックスを統合 (重複削除) - torch.unique_consecutive 使用"""
        if not candidate_indices_groups: # 空の場合の処理を追加
            return torch.empty((candidate_indices_groups[0].size(0), candidate_indices_groups[0].size(1), 0), dtype=torch.long, device=candidate_indices_groups[0].device)

        merged_candidates = torch.cat(candidate_indices_groups, dim=-1)
        merged_candidates, _ = torch.sort(merged_candidates) # Sort for unique_consecutive
        return torch.unique_consecutive(merged_candidates, dim=-1)

    def select_top_k_candidates(self, combined_candidates_indices: torch.Tensor, query_up_group: torch.Tensor, key_up_group: torch.Tensor, batch_idx: int, query_idx: int) -> torch.Tensor:
        """Top-k 候補を選択"""
        if len(combined_candidates_indices) <= self.config.k_max:
            return combined_candidates_indices

        # 類似度計算をベクトル化
        similarities = torch.matmul(key_up_group[batch_idx, combined_candidates_indices], query_up_group[batch_idx, query_idx].unsqueeze(0).transpose(0, 1)).squeeze(1)
        _, top_indices = torch.topk(similarities, self.config.k_max)
        return combined_candidates_indices[top_indices]


    def _process_dimension_group(self, query_up_group: torch.Tensor, key_up_group: torch.Tensor, head_idx: int, group_idx: int, batch_size: int, seq_len: int, tries_built_for_batch: dict) -> torch.Tensor:
        """次元グループに対する候補探索処理 (内部関数)"""
        lsh_matches = self.get_lsh_matches(query_up_group, key_up_group, head_idx, group_idx) # [batch_size, seq_len, seq_len]
        initial_candidates_lists = self._get_initial_candidates_wu_manber_group(query_up_group, key_up_group, head_idx) # [seq_len, List[int]] (batch_size=1前提)
        trie_candidate_lists = self._get_trie_candidates_group(batch_size - 1, query_up_group, key_up_group, head_idx, initial_candidates_lists, tries_built_for_batch) # [seq_len, List[int]] (batch_size=1前提)

        candidates_group = torch.full((batch_size, seq_len, self.config.k_max), -1, dtype=torch.long, device=query_up_group.device)

        for batch_idx in range(batch_size): # Potential Parallelization (vmap or custom kernel): Batch Loop
            for query_idx in range(seq_len): # Potential Parallelization (vmap or custom kernel): Sequence Length Loop
                matched_indices_tensor = lsh_matches[batch_idx, query_idx].nonzero(as_tuple=False).squeeze(1) # Tensor of matched indices
                if matched_indices_tensor.numel() > 0:
                    trie_candidates = trie_candidate_lists[query_idx]
                    trie_candidates_tensor = torch.tensor(trie_candidates, dtype=torch.long, device=query_up_group.device)

                    # Tensor operations for intersection using torch.isin
                    combined_candidates_indices = trie_candidates_tensor[torch.isin(trie_candidates_tensor, matched_indices_tensor)]
                    if combined_candidates_indices.numel() > 0: # 候補が空でない場合のみTop-k選択
                        selected_top_k_indices = self.select_top_k_candidates(combined_candidates_indices, query_up_group, key_up_group, batch_idx, query_idx)
                        candidate_count = min(len(selected_top_k_indices), self.config.k_max)
                        candidates_group[batch_idx, query_idx, :candidate_count] = selected_top_k_indices[:candidate_count]

        return candidates_group


    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        batch_size, seq_len, _ = query_up.size()
        query_up_groups = self.split_features_by_dim_groups(query_up)
        key_up_groups = self.split_features_by_dim_groups(key_up)

        candidate_indices_groups = []
        tries_built_for_batch = {} # バッチごとに Trie が構築されたかを追跡する辞書

        # Potential Parallelization (vmap or custom kernel): Dimension Group Loop
        for group_idx, (query_up_group, key_up_group) in enumerate(zip(query_up_groups, key_up_groups)):
            candidates_group = self._process_dimension_group(query_up_group, key_up_group, head_idx, group_idx, batch_size, seq_len, tries_built_for_batch)
            candidate_indices_groups.append(candidates_group)

        if not candidate_indices_groups: # 次元分割がない場合などの空リスト対策
            return torch.full((batch_size, seq_len, self.config.k_max), -1, dtype=torch.long, device=query_up.device)

        candidates = self.merge_candidate_indices(candidate_indices_groups)
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
                LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes) # n_lsh_hashes を渡す
                for dim in config.hyper_cuts_dim_groups
            ])
            for _ in range(config.n_heads)
        ]) if config.hyper_cuts_dim_groups else nn.ModuleList([
            nn.ModuleList([LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)]) for _ in range(config.n_heads) # n_lsh_hashes を渡す
        ])

        self.tries_list = nn.ModuleList([Trie(config.stride) for _ in range(config.n_heads)])
        self.candidate_finder = CandidateFinder(config, self.tries_list, self.lsh_tables_list)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        query_down = self.query_down_proj(query)
        key_value_down = self.key_value_down_proj(key)
        head_outputs = []

        for head_idx in range(self.config.n_heads):
            query_up = self.query_up_projs[head_idx](query_down)
            key_up = self.key_up_projs[head_idx](key_value_down)
            candidates = self.candidate_finder(query_up, key_up, head_idx)

            head_output = torch.zeros(batch_size, seq_len, self.config.d_model, device=query.device)
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    valid_candidate_indices = candidates[batch_idx, seq_idx][candidates[batch_idx, seq_idx] != -1]
                    if valid_candidate_indices.numel() > 0:
                        q_up = query_up[batch_idx, seq_idx].unsqueeze(0)
                        k_up = key_up[batch_idx, valid_candidate_indices]
                        scores = q_up.matmul(k_up.transpose(-2, -1)) / math.sqrt(self.config.d_key)

                        if mask is not None:
                            mask_slice = mask[batch_idx, seq_idx, :seq_len]
                            scores = scores.masked_fill(~mask_slice[valid_candidate_indices].unsqueeze(0), float('-inf'))

                        attn_weights = F.softmax(scores, dim=-1)
                        v_up = self.value_up_projs[head_idx](key_value_down[batch_idx, valid_candidate_indices])
                        head_output[batch_idx, seq_idx] = attn_weights.matmul(v_up).squeeze(0)
            head_outputs.append(head_output)

        concat_output = torch.cat(head_outputs, dim=-1)
        return self.output_proj(concat_output)


def example_usage():
    config = FastAttentionConfig(d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
                                  rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
                                  lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
                                  hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4) # n_lsh_hashes を追加
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastAttention(config).to(device)
    batch_size, seq_len = 2, 128
    query = torch.randn(batch_size, seq_len, config.d_model).to(device)
    key = torch.randn(batch_size, seq_len, config.d_model).to(device)
    value = torch.randn(batch_size, seq_len, config.d_model).to(device)
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool).to(device)
    output = model(query, key, value, mask)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    example_usage()
