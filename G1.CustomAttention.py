import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math
import collections  # For OrderedDict for LRU cache

@dataclass
class FastAttentionConfig:
    """FastAttention Configuration (Hyper-Cuts Dimension Splitting Support)"""
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
    n_lsh_hashes: int = 4  # Number of LSH hash functions per table
    n_lsh_tables: int = 1   # Number of LSH tables per head (for ensemble - default 1)
    trie_cache_max_size: int = 4 # Max size for Trie LRU cache


class LowRankLinear(nn.Module):
    """Low-Rank Linear Layer"""
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.u_weight = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.v_weight = nn.Parameter(torch.randn(rank, out_features) / math.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.u_weight).matmul(self.v_weight)

class RandomFourierFeatures(nn.Module):
    """Random Fourier Features"""
    def __init__(self, input_dim: int, rff_dim: int):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi, requires_grad=False)
        self.scale = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = x.matmul(self.omega) + self.bias
        return torch.cos(projection) * self.scale

class LSHTable(nn.Module):
    """LSH Table with Multiple Hash Functions"""
    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes
        self.random_vectors = nn.Parameter(torch.randn(dim, n_hashes), requires_grad=False) # Multiple hash functions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hash(x)

    def hash(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.matmul(self.random_vectors)
        return torch.floor(proj / self.bandwidth) % self.n_buckets

class Trie(nn.Module):
    """Trie Tree"""
    def __init__(self, stride: int):
        super().__init__()
        self.root_node = {}
        self.stride_len = stride
        self._indices_key = '_indices' # More descriptive key for indices

    def insert(self, binary_vector: torch.Tensor, index: int):
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())
            if prefix not in current_node:
                current_node[prefix] = {}
            current_node = current_node[prefix]
        if self._indices_key not in current_node:
            current_node[self._indices_key] = []
        current_node[self._indices_key].append(index)

    def search(self, binary_vector: torch.Tensor) -> List[int]:
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())
            if prefix not in current_node:
                return []
            current_node = current_node[prefix]
        return current_node.get(self._indices_key, [])

class CandidateFinder(nn.Module):
    """Candidate Finder (Wu-Manber + Trie + Hyper-Cuts) with Trie Caching and vmap (Conceptual)"""
    def __init__(self, config: FastAttentionConfig, tries: List[Trie], lsh_tables: nn.ModuleList):
        super().__init__()
        self.config = config
        self.tries = tries
        self.lsh_tables = lsh_tables
        self.wu_manber_prefix_len = config.wu_manber_prefix_len
        self.hyper_cuts_dim_groups = config.hyper_cuts_dim_groups
        self.trie_cache = collections.OrderedDict() # LRU Cache for Tries
        self.trie_cache_max_size = config.trie_cache_max_size

    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        """Splits features into dimension groups based on hyper-Cuts config."""
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
        """LSH Matching for a dimension group, considering multiple hash tables and functions."""
        lsh_tables_for_group = self.lsh_tables[head_idx][group_idx] # Now list of LSH tables

        # Combine results from multiple LSH tables (OR condition across tables)
        combined_matches = torch.zeros((query_up_group.size(0), query_up_group.size(1), key_up_group.size(1)), dtype=torch.bool, device=query_up_group.device)
        for lsh_table in lsh_tables_for_group:
            query_hashes = lsh_table(query_up_group) # [batch_size, seq_len, n_lsh_hashes]
            key_hashes = lsh_table(key_up_group)     # [batch_size, seq_len, n_lsh_hashes]
            matches = torch.zeros((query_hashes.size(0), query_hashes.size(1), key_hashes.size(1)), dtype=torch.bool, device=query_hashes.device)
            for hash_idx in range(self.config.n_lsh_hashes):
                matches = matches | (query_hashes[:, :, None, hash_idx] == key_hashes[:, None, :, hash_idx]) # Compare each hash function
            combined_matches = combined_matches | matches # OR across tables
        return combined_matches

    def build_wu_manber_hash_table(self, key_binary_batch: torch.Tensor, prefix_len: int) -> dict:
        """Builds Wu-Manber hash table."""
        hash_table = {}
        seq_len = key_binary_batch.size(1)
        for key_index in range(seq_len):
            key_vec = key_binary_batch[0, key_index].cpu() # Still using CPU for hash table keys
            prefix = tuple(key_vec[:prefix_len].tolist())
            if prefix not in hash_table:
                hash_table[prefix] = []
            hash_table[prefix].append(key_index)
        return hash_table

    def wu_manber_search(self, query_binary_vec: torch.Tensor, wu_manber_hash_table: dict, prefix_len: int) -> List[int]:
        """Wu-Manber search for candidate indices."""
        prefix = tuple(query_binary_vec[:prefix_len].tolist())
        return wu_manber_hash_table.get(prefix, [])

    def _get_initial_candidates_wu_manber_group(self, query_up_group: torch.Tensor, key_up_group: torch.Tensor, head_idx: int) -> List[List[int]]:
        """Get initial candidates using Wu-Manber per dimension group."""
        key_binary_batch = self.binary_quantize(key_up_group)
        wu_manber_hash_table = self.build_wu_manber_hash_table(key_binary_batch, self.config.wu_manber_prefix_len)
        seq_len = query_up_group.size(1)
        initial_candidates_list = []
        for query_index in range(seq_len):
            query_vec_binary = self.binary_quantize(query_up_group[0, query_index]).cpu() # Batch index 0 is assumed for hash table build
            initial_candidates_list.append(self.wu_manber_search(query_vec_binary, wu_manber_hash_table, self.config.wu_manber_prefix_len))
        return initial_candidates_list

    def _get_trie_group(self, key_up_group: torch.Tensor, head_idx: int) -> Trie:
        """Gets Trie from LRU cache, building and caching if necessary."""
        key_hash = hash(key_up_group.detach().cpu().numpy().tobytes()) # Detach before numpy() and hashing

        if key_hash in self.trie_cache:
            trie = self.trie_cache.pop(key_hash) # LRU: Move to end (most recently used)
            self.trie_cache[key_hash] = trie
            return trie

        trie = self.tries[head_idx]
        trie.root_node = {} # Clear Trie
        key_binary_batch = self.binary_quantize(key_up_group)
        seq_len = key_up_group.size(1)
        for key_index in range(seq_len):
            trie.insert(key_binary_batch[0, key_index].cpu(), key_index)

        self._add_to_trie_cache(key_hash, trie) # Add to LRU cache
        return trie

    def _add_to_trie_cache(self, key_hash, trie):
        """Adds Trie to LRU cache, evicting oldest if cache is full."""
        if key_hash in self.trie_cache:
            self.trie_cache.move_to_end(key_hash) # LRU update if already present
            return

        self.trie_cache[key_hash] = trie
        if len(self.trie_cache) > self.trie_cache_max_size:
            self.trie_cache.popitem(last=False) # LRU eviction (remove oldest)


    def _get_trie_candidates_group(self, batch_idx: int, query_up_group: torch.Tensor, key_up_group: torch.Tensor, head_idx: int, initial_candidates_lists: List[List[int]]) -> List[List[int]]:
        """Refine candidates using Trie per dimension group - now using cached Trie."""
        trie = self._get_trie_group(key_up_group, head_idx) # Get Trie from cache

        trie_candidates_list = []
        seq_len = query_up_group.size(1)
        for query_index in range(seq_len):
            query_vec_binary = self.binary_quantize(query_up_group[batch_idx, query_index]).cpu()
            trie_candidates = trie.search(query_vec_binary)
            trie_candidates_list.append(trie_candidates)
        return trie_candidates_list


    def merge_candidate_indices(self, candidate_indices_groups: List[torch.Tensor]) -> torch.Tensor:
        """Merges candidate indices from dimension groups (deduplication)."""
        merged_candidates = torch.cat(candidate_indices_groups, dim=-1)
        merged_candidates, _ = torch.sort(merged_candidates) # Sort for unique_consecutive
        return torch.unique_consecutive(merged_candidates, dim=-1)

    def select_top_k_candidates(self, combined_candidates: torch.Tensor, query_up: torch.Tensor, key_up: torch.Tensor, batch_idx: int, query_idx: int) -> torch.Tensor:
        """Selects top-k candidates based on similarity."""
        if len(combined_candidates) <= self.config.k_max:
            return combined_candidates
        similarities = key_up[batch_idx, combined_candidates].matmul(query_up[batch_idx, query_idx])
        _, top_indices = torch.topk(similarities, self.config.k_max)
        return combined_candidates[top_indices]

    def _process_dimension_group(self, query_up_group: torch.Tensor, key_up_group: torch.Tensor, head_idx: int, group_idx: int, batch_size: int, seq_len: int) -> torch.Tensor:
        """Processes candidate search for a single dimension group.  Potentially vectorizable with vmap."""
        lsh_matches = self.get_lsh_matches(query_up_group, key_up_group, head_idx, group_idx) # [batch_size, seq_len, key_seq_len]
        initial_candidates_lists = self._get_initial_candidates_wu_manber_group(query_up_group, key_up_group, head_idx) # List[List[int]]
        trie_candidate_lists = self._get_trie_candidates_group(batch_size - 1, query_up_group, key_up_group, head_idx, initial_candidates_lists) # List[List[int]]

        candidates_group = torch.full((batch_size, seq_len, self.config.k_max), -1, dtype=torch.long, device=query_up_group.device)

        # --- Potential vmap Vectorization Point ---
        # The loops below are still present.  For true vmap vectorization, significant refactoring is needed.
        # The following loop structure is kept for conceptual clarity and to highlight where vmap *could* be applied.
        for batch_idx in range(batch_size):
            for query_idx in range(seq_len):
                matched_indices_lsh = lsh_matches[batch_idx, query_idx].nonzero(as_tuple=True)[0] # LSH matched indices
                if len(matched_indices_lsh) > 0:
                    trie_candidates = trie_candidate_lists[query_idx]
                    # Efficient intersection using tensor operations instead of sets:
                    trie_candidate_tensor = torch.tensor(trie_candidates, dtype=torch.long, device=query_up_group.device) if trie_candidates else torch.empty(0, dtype=torch.long, device=query_up_group.device)
                    # Using torch.isin for efficient intersection (candidates present in both LSH and Trie results)
                    combined_candidate_indices_mask = torch.isin(trie_candidate_tensor, matched_indices_lsh)
                    combined_candidate_indices = trie_candidate_tensor[combined_candidate_indices_mask]

                    selected_top_k_indices = self.select_top_k_candidates(combined_candidate_indices, query_up_group, key_up_group, batch_idx, query_idx)
                    candidate_count = min(len(selected_top_k_indices), self.config.k_max)
                    candidates_group[batch_idx, query_idx, :candidate_count] = selected_top_k_indices[:candidate_count]
        return candidates_group


    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        batch_size, seq_len, _ = query_up.size()
        query_up_groups = self.split_features_by_dim_groups(query_up)
        key_up_groups = self.split_features_by_dim_groups(key_up)

        candidate_indices_groups = []
        # Potential Parallelization Point: Dimension group processing could be parallelized.
        for group_idx, (query_up_group, key_up_group) in enumerate(zip(query_up_groups, key_up_groups)):
            candidates_group = self._process_dimension_group(query_up_group, key_up_group, head_idx, group_idx, batch_size, seq_len)
            candidate_indices_groups.append(candidates_group)

        candidates = self.merge_candidate_indices(candidate_indices_groups)
        return candidates[:, :, :self.config.k_max]


class FastAttention(nn.Module):
    """Fast Attention Mechanism (Wu-Manber + Trie + Hyper-Cuts) with Trie Caching and LSH Ensemble"""
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
                nn.ModuleList([ # List of LSH tables for ensemble
                    LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes) # Pass n_lsh_hashes
                    for _ in range(config.n_lsh_tables) # Multiple LSH tables for ensemble
                ])
                for dim in config.hyper_cuts_dim_groups
            ])
            for _ in range(config.n_heads)
        ]) if config.hyper_cuts_dim_groups else nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([ # List of LSH tables for ensemble
                    LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
                    for _ in range(config.n_lsh_tables) # Multiple LSH tables for ensemble
                ])
            ]) for _ in range(config.n_heads)
        ])

        self.tries_list = nn.ModuleList([Trie(config.stride) for _ in range(config.n_heads)])
        self.candidate_finder = CandidateFinder(config, self.tries_list, self.lsh_tables_list)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        query_down = self.query_down_proj(query)
        key_value_down = self.key_value_down_proj(key)
        head_outputs = []

        # Trie building is now handled within CandidateFinder._get_trie_group using LRU Cache.
        # No explicit Trie building loop here anymore.

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
                            mask_slice = mask[batch_idx, seq_idx, :seq_len] # Assuming mask is [B, Q_len, K_len]
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
                                  hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4,
                                  n_lsh_tables=2, trie_cache_max_size=4) # Added n_lsh_hashes, n_lsh_tables, trie_cache_max_size
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
