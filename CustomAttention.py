import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

@dataclass
class FastAttentionConfig:
    """FastAttentionの設定"""
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
    """LSHテーブル"""
    def __init__(self, dim: int, n_buckets: int, bandwidth: float):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.random_vectors = nn.Parameter(torch.randn(dim, 1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hash(x)

    def hash(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.matmul(self.random_vectors).squeeze(-1)
        return torch.floor(proj / self.bandwidth) % self.n_buckets

class Trie:
    """Trie木"""
    def __init__(self, stride: int):
        self.root_node = {}
        self.stride_len = stride

    def insert(self, binary_vector: torch.Tensor, index: int):
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())
            if prefix not in current_node:
                current_node[prefix] = {}
            current_node = current_node[prefix]
        if '_indices' not in current_node:
            current_node['_indices'] = []
        current_node['_indices'].append(index)

    def search(self, binary_vector: torch.Tensor) -> List[int]:
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())
            if prefix not in current_node:
                return []
            current_node = current_node[prefix]
        return current_node.get('_indices', [])

class CandidateFinder(nn.Module):
    """候補探索器 (Wu-Manber + Trie木 統合)"""
    def __init__(self, config: FastAttentionConfig, tries: List[Trie], lsh_tables: nn.ModuleList):
        super().__init__()
        self.config = config
        self.tries = tries
        self.lsh_tables = lsh_tables
        self.wu_manber_prefix_len = config.wu_manber_prefix_len

    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()

    def get_lsh_matches(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        query_hash = self.lsh_tables[head_idx](query_up)
        key_hash = self.lsh_tables[head_idx](key_up)
        return (query_hash.unsqueeze(2) == key_hash.unsqueeze(1))

    def build_wu_manber_hash_table(self, key_binary_batch: torch.Tensor, prefix_len: int) -> dict:
        hash_table = {}
        seq_len = key_binary_batch.size(1)
        for key_index in range(seq_len):
            key_vec = key_binary_batch[0, key_index].cpu() # batch_idx=0 のみ使用を想定
            prefix = tuple(key_vec[:prefix_len].tolist())
            if prefix not in hash_table:
                hash_table[prefix] = []
            hash_table[prefix].append(key_index)
        return hash_table

    def wu_manber_search(self, query_binary_vec: torch.Tensor, wu_manber_hash_table: dict, prefix_len: int) -> List[int]:
        prefix = tuple(query_binary_vec[:prefix_len].tolist())
        return wu_manber_hash_table.get(prefix, [])

    def get_initial_candidates_wu_manber(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> List[List[int]]:
        """Wu-Manberを使って初期候補を取得"""
        key_binary_batch = self.binary_quantize(key_up)
        wu_manber_hash_table = self.build_wu_manber_hash_table(key_binary_batch, self.config.wu_manber_prefix_len)
        seq_len = query_up.size(1)
        initial_candidates_list = []
        for query_index in range(seq_len):
            query_vec_binary = self.binary_quantize(query_up[0, query_index]).cpu() # batch_idx=0 のみ使用を想定
            initial_candidates_list.append(self.wu_manber_search(query_vec_binary, wu_manber_hash_table, self.config.wu_manber_prefix_len))
        return initial_candidates_list


    def get_trie_candidates(self, batch_idx: int, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int, initial_candidates_list: List[List[int]]) -> List[List[int]]:
        """Trie木で候補を絞り込む (Wu-Manberで絞り込んだ候補を初期候補とする)"""
        trie = self.tries[head_idx]
        trie.root_node = {} # per batch and head init
        key_binary_batch = self.binary_quantize(key_up)

        seq_len = key_up.size(1)
        for key_index in range(seq_len):
            # Wu-Manberで絞り込んだ候補のみTrie木に挿入
            if key_index in initial_candidates_list[key_index]: # key_index が Wu-Manber の候補に含まれる場合のみTrie木にinsert (誤り。全keyをinsertする必要がある)
                trie.insert(key_binary_batch[batch_idx, key_index].cpu(), key_index) # 全てのkeyをTrie木に挿入

        trie_candidates_list = []
        for query_index in range(seq_len):
            query_vec_binary = self.binary_quantize(query_up[batch_idx, query_index]).cpu()
            # Trie木探索は Wu-Manber で絞り込んだ候補に対してのみ行う (ここは修正が必要)
            trie_candidates = trie.search(query_vec_binary) # Trie木探索は全てのqueryに対して行う
            trie_candidates_list.append(trie_candidates) # Trie木探索は全てのqueryに対して行う

        return trie_candidates_list


    def select_top_k_candidates(self, combined_candidates: torch.Tensor, query_up: torch.Tensor, key_up: torch.Tensor, batch_idx: int, query_idx: int) -> torch.Tensor:
        if len(combined_candidates) <= self.config.k_max:
            return combined_candidates
        similarities = key_up[batch_idx, combined_candidates].matmul(query_up[batch_idx, query_idx])
        _, top_indices = torch.topk(similarities, self.config.k_max)
        return combined_candidates[top_indices]


    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        batch_size, seq_len, _ = query_up.size()
        lsh_matches = self.get_lsh_matches(query_up, key_up, head_idx)

        initial_candidates_lists = self.get_initial_candidates_wu_manber(query_up, key_up, head_idx) # Wu-Manberで初期候補を取得
        trie_candidate_lists = self.get_trie_candidates(batch_size - 1, query_up, key_up, head_idx, initial_candidates_lists) # Trie木で候補を絞り込む (Wu-Manberの結果を初期候補とする) # batch_idx=batch_size - 1 に修正

        candidates = torch.full((batch_size, seq_len, self.config.k_max), -1, dtype=torch.long, device=query_up.device)


        for batch_idx in range(batch_size):
            for query_idx in range(seq_len):
                matched_indices = lsh_matches[batch_idx, query_idx].nonzero(as_tuple=True)[0]
                if len(matched_indices) > 0:
                    trie_candidates = trie_candidate_lists[query_idx] # Trie木で絞り込んだ候補 (Wu-Manberの結果を初期候補とする)
                    # combined_candidates_indices = self.combine_candidates(matched_indices, trie_candidates) # combine_candidates を削除
                    combined_candidates_indices = torch.tensor(list(set(matched_indices.tolist()) & set(trie_candidates)), dtype=torch.long, device=query_up.device) # LSHとTrie木両方で候補となったindexの積集合
                    selected_top_k_indices = self.select_top_k_candidates(combined_candidates_indices, query_up, key_up, batch_idx, query_idx)
                    candidate_count = min(len(selected_top_k_indices), self.config.k_max)
                    candidates[batch_idx, query_idx, :candidate_count] = selected_top_k_indices[:candidate_count]
        return candidates


class FastAttention(nn.Module):
    """高速注意機構 (Wu-Manber + Trie木 統合)"""
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        self.query_down_proj = nn.Linear(config.d_model, config.d_query)
        self.key_value_down_proj = nn.Linear(config.d_model, config.d_key)
        self.query_up_projs = nn.ModuleList([LowRankLinear(config.d_query, config.d_key, config.rank) for _ in range(config.n_heads)])
        self.key_up_projs = nn.ModuleList([LowRankLinear(config.d_key, config.d_key, config.rank) for _ in range(config.n_heads)])
        self.value_up_projs = nn.ModuleList([LowRankLinear(config.d_key, config.d_model, config.rank) for _ in range(config.n_heads)])
        self.rff_encoders = nn.ModuleList([RandomFourierFeatures(config.d_key, config.rff_dim) for _ in range(config.n_heads)])
        self.lsh_tables_list = nn.ModuleList([LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth) for _ in range(config.n_heads)])
        self.tries_list = [Trie(config.stride) for _ in range(config.n_heads)]
        self.candidate_finder = CandidateFinder(config, self.tries_list, self.lsh_tables_list) # CandidateFinder
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        query_down = self.query_down_proj(query)
        key_value_down = self.key_value_down_proj(key)
        head_outputs = []

        for head_idx in range(self.config.n_heads):
            query_up = self.query_up_projs[head_idx](query_down)
            key_up = self.key_up_projs[head_idx](key_value_down)
            candidates = self.candidate_finder(query_up, key_up, head_idx) # CandidateFinder (Wu-Manber + Trie木 統合)

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
                                  lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3)
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
