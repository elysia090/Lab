import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable, Any, Union
from torch.utils.tensorboard import SummaryWriter
import logging
import time
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

##############################################
# Low-level Helpers - Improved
##############################################

def optimized_sqrt(n: int) -> float:
    """
    Optimized square root for powers of 2.

    Args:
        n (int): Input integer.

    Returns:
        float: Square root of n, optimized for powers of 2.

    Raises:
        ValueError: If input is not positive.
    """
    if n <= 0:
        raise ValueError("Input must be positive")
    if n & (n - 1) == 0:
        k = n.bit_length() - 1
        return 2 ** (k / 2)
    return math.sqrt(n)

def fma(a: float, b: float, c: float) -> float:
    """
    Fused multiply-add with fallback to standard operations.

    Args:
        a (float): First factor.
        b (float): Second factor.
        c (float): Addend.

    Returns:
        float: Result of fused multiply-add operation (a*b + c).
    """
    try:
        return math.fma(a, b, c)
    except AttributeError:
        return a * b + c

def validate_tensor_dimensions(tensor: torch.Tensor, name: str, expected_dims: int):
    """
    Validates tensor dimensions to provide clear error messages.

    Args:
        tensor (torch.Tensor): Input tensor to validate.
        name (str): Name of the tensor for error messages.
        expected_dims (int): Expected number of dimensions.

    Raises:
        ValueError: If tensor dimensions do not match expected dimensions.
    """
    if tensor.dim() != expected_dims:
        raise ValueError(f"Expected {name} to have {expected_dims} dimensions, got {tensor.dim()}")

def check_for_nan(tensor: torch.Tensor, tensor_name: str):
    """
    Checks if a tensor contains NaN values and raises a ValueError if it does.

    Args:
        tensor (torch.Tensor): Input tensor to check.
        tensor_name (str): Name of the tensor for error messages.

    Raises:
        ValueError: If tensor contains NaN values.
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {tensor_name}!")

def check_for_inf(tensor: torch.Tensor, tensor_name: str):
    """
    Checks if a tensor contains inf values and raises a ValueError if it does.
    """
    if torch.isinf(tensor).any():
        raise ValueError(f"inf detected in {tensor_name}!")
##############################################
# Fast Attention Components - Improved
##############################################

@dataclass
class FastAttentionConfig:
    """
    Configuration for FastAttention with validation.

    Attributes:
        # ... (existing attributes)
        dtype: torch.dtype = torch.float32
    """
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
    use_rff: bool = True
    dtype: torch.dtype = torch.float32  # dtype field as a dataclass attribute

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_model <= 0 or self.d_key <= 0 or self.d_query <= 0:
            raise ValueError("Dimensions must be positive")
        if self.n_heads <= 0:
            raise ValueError("Number of heads must be positive")
        if self.rank <= 0:
            raise ValueError("Rank must be positive")
        if self.hyper_cuts_dim_groups is not None:
            if sum(self.hyper_cuts_dim_groups) != self.lsh_key_dim:
                raise ValueError(f"Sum of hyper_cuts_dim_groups ({sum(self.hyper_cuts_dim_groups)}) "
                                f"must equal lsh_key_dim ({self.lsh_key_dim})")

class LowRankLinear(nn.Module):
    """
    Improved LowRankLinear with better initialization and caching.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        std = 1.0 / math.sqrt(rank)
        self.u_weight = nn.Parameter(torch.randn(in_features, rank, dtype=dtype) * std)
        self.v_weight = nn.Parameter(torch.randn(rank, out_features, dtype=dtype) * std)
        self.register_buffer('composed_weight', None)
        self.needs_composition = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank <= min(self.in_features, self.out_features) // 4:
            if self.needs_composition or self.composed_weight is None:
                self.composed_weight = torch.matmul(self.u_weight, self.v_weight)
                self.needs_composition = False
            return torch.matmul(x, self.composed_weight)
        return torch.matmul(torch.matmul(x, self.u_weight), self.v_weight)

    def train(self, mode: bool = True):
        if mode and not self.training:
            self.needs_composition = True
        return super().train(mode)

class RandomFourierFeatures(nn.Module):
    """
    Improved RandomFourierFeatures with normalized projections and JIT compatibility.
    """
    def __init__(self, input_dim: int, rff_dim: int, dtype=torch.float32):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim, dtype=dtype) / math.sqrt(input_dim),
                                 requires_grad=False)
        self.bias = nn.Parameter(torch.rand(rff_dim, dtype=dtype) * 2 * math.pi, requires_grad=False)
        self.scale = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])
        projection = torch.addmm(self.bias, x, self.omega)
        result = torch.cos(projection) * self.scale
        check_for_nan(result, "RFF output")
        check_for_inf(result, "RFF output")
        if len(orig_shape) > 2:
            new_shape = list(orig_shape[:-1]) + [self.omega.size(1)]
            result = result.view(*new_shape)
        return result

class LSHTable(nn.Module):
    """
    Improved LSHTable with better hashing and optional seed control.
    """
    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int, seed: Optional[int] = None, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes

        if seed is not None:
            torch.manual_seed(seed)

        random_vectors = self._init_quasi_orthogonal_vectors(dim, n_hashes, dtype=dtype)
        self.register_buffer("random_vectors", random_vectors)
        self.register_buffer("collision_count", torch.zeros(1))

    def _init_quasi_orthogonal_vectors(self, dim: int, n_hashes: int, dtype=torch.float32) -> torch.Tensor:
        vectors = torch.randn(dim, n_hashes, dtype=dtype)

        if n_hashes <= dim and n_hashes <= 10:
            for i in range(1, n_hashes):
                for j in range(i):
                    proj = torch.sum(vectors[:, i] * vectors[:, j]) / torch.sum(vectors[:, j] ** 2)
                    vectors[:, i] = vectors[:, i] - proj * vectors[:, j]
        return vectors / torch.norm(vectors, dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_dimensions(x, "input", x.dim())
        proj = x.matmul(self.random_vectors)
        hashed = torch.floor(proj / self.bandwidth) % self.n_buckets
        check_for_nan(hashed, "LSH hashed output")
        check_for_inf(hashed, "LSH hashed output")

        if self.training:
            unique_hashes = torch.unique(hashed.reshape(-1, self.n_hashes), dim=0).shape[0]
            total_hashes = hashed.reshape(-1, self.n_hashes).shape[0]
            expected_unique = min(total_hashes, self.n_buckets ** self.n_hashes)
            if expected_unique > 0 and total_hashes > 0:
                self.collision_count[0] = 1.0 - (unique_hashes / expected_unique)
        return hashed

_TRIE_INDICES_KEY = '_indices'

class Trie(nn.Module):
    """
    Improved Trie with memory-efficient storage and faster lookups.
    """
    def __init__(self, stride: int):
        super().__init__()
        self.root_node: Dict = {}
        self.stride_len = stride
        self.max_depth = 0
        self.node_count = 0

    def insert(self, binary_vector: torch.Tensor, index: int) -> None:
        if binary_vector.dim() != 1:
            raise ValueError(f"Expected 1D binary_vector, got {binary_vector.dim()}D")
        current_node = self.root_node
        depth = 0
        for i in range(0, len(binary_vector), self.stride_len):
            depth += 1
            end_idx = min(i + self.stride_len, len(binary_vector))
            prefix = tuple(binary_vector[i:end_idx].tolist())
            if prefix not in current_node:
                current_node[prefix] = {}
                self.node_count += 1
            current_node = current_node[prefix]
        if _TRIE_INDICES_KEY not in current_node:
            current_node[_TRIE_INDICES_KEY] = []
        current_node[_TRIE_INDICES_KEY].append(index)
        self.max_depth = max(self.max_depth, depth)

    def search(self, binary_vector: torch.Tensor) -> List[int]:
        if binary_vector.dim() != 1:
            raise ValueError(f"Expected 1D binary_vector, got {binary_vector.dim()}D")
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            end_idx = min(i + self.stride_len, len(binary_vector))
            prefix = tuple(binary_vector[i:end_idx].tolist())
            if prefix not in current_node:
                return []
            current_node = current_node[prefix]
        return current_node.get(_TRIE_INDICES_KEY, [])

    def clear(self) -> None:
        self.root_node.clear()
        self.max_depth = 0
        self.node_count = 0

class TrieCache(nn.Module):
    """
    Cache of Tries for efficient reuse.
    """
    def __init__(self, stride: int, max_cache_size: int = 16):
        super().__init__()
        self.stride = stride
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()

    def get_trie(self, data_hash: int) -> Tuple[Trie, bool]:
        if data_hash in self.cache:
            trie = self.cache.pop(data_hash)
            self.cache[data_hash] = trie
            return trie, True
        trie = Trie(self.stride)
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)
        self.cache[data_hash] = trie
        return trie, False

class CandidateFinder(nn.Module):
    """
    Improved CandidateFinder with caching and parallel processing.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        self.wu_manber_prefix_len = config.wu_manber_prefix_len
        self.hyper_cuts_dim_groups = config.hyper_cuts_dim_groups
        self.lsh_tables = nn.ModuleList()
        for _ in range(config.n_heads):
            if config.hyper_cuts_dim_groups:
                head_tables = nn.ModuleList()
                for dim in config.hyper_cuts_dim_groups:
                    head_tables.append(
                        LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes, dtype=config.dtype)
                    )
                self.lsh_tables.append(head_tables)
            else:
                self.lsh_tables.append(nn.ModuleList([
                    LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes, dtype=config.dtype)
                ]))
        self.trie_cache = TrieCache(config.stride)
        self.wu_manber_cache = OrderedDict()
        self.wu_manber_cache_hits = 0
        self.wu_manber_cache_misses = 0

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

    def _build_wu_manber_hash_table(self, key_bin: torch.Tensor) -> Dict[tuple, List[int]]:
        key_hash = hash(tuple(key_bin[:, :self.config.wu_manber_prefix_len].flatten().tolist()))
        if key_hash in self.wu_manber_cache:
            self.wu_manber_cache_hits += 1
            table = self.wu_manber_cache.pop(key_hash)
            self.wu_manber_cache[key_hash] = table
            return table
        self.wu_manber_cache_misses += 1
        table: Dict[tuple, List[int]] = {}
        L = key_bin.size(0)
        prefixes = [tuple(key_bin[i, :self.config.wu_manber_prefix_len].tolist()) for i in range(L)]
        for i, prefix in enumerate(prefixes):
            table.setdefault(prefix, []).append(i)
        self.wu_manber_cache[key_hash] = table
        if len(self.wu_manber_cache) > 1000:
            self.wu_manber_cache.popitem(last=False)
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
            batch_list = [self._wu_manber_search(query_bin[b, i], table) for i in range(L)]
            cand_lists.append(batch_list)
        return cand_lists

    def _get_trie_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> List[List[List[int]]]:
        B, L, _ = key_grp.size()
        cand_lists = []
        query_bin_all = self.binary_quantize(query_grp)
        for b in range(B):
            key_data = key_grp[b].detach()
            data_hash = hash(tuple(key_data[0, :50].tolist())) + hash(str(head_idx)) + b
            trie, cache_hit = self.trie_cache.get_trie(data_hash)
            if not cache_hit:
                key_bin = self.binary_quantize(key_grp[b])
                for i in range(L):
                    trie.insert(key_bin[i], i)
            query_bin = query_bin_all[b]
            batch_list = []
            chunk_size = 32
            for chunk_start in range(0, L, chunk_size):
                chunk_end = min(chunk_start + chunk_size, L)
                chunk_results = [trie.search(query_bin[i]) for i in range(chunk_start, chunk_end)]
                batch_list.extend(chunk_results)
            cand_lists.append(batch_list)
        return cand_lists

    def merge_candidate_indices_groups(self, cand_tensors: List[torch.Tensor]) -> torch.Tensor:
        if not cand_tensors:
            return None
        merged = torch.cat(cand_tensors, dim=-1)
        merged, _ = torch.sort(merged, dim=-1)
        return torch.unique(merged, dim=-1)

    def _process_dimension_group_candidates(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> torch.Tensor:
        B, L, _ = query_grp.size()
        device = query_grp.device
        wu_cands = self._get_wu_manber_candidates_group(query_grp, key_grp)
        trie_cands = self._get_trie_candidates_group(query_grp, key_grp, head_idx)
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=device)
        for b in range(B):
            for i in range(L):
                common = list(set(wu_cands[b][i]) & set(trie_cands[b][i]))
                if common:
                    common_tensor = torch.tensor(common, dtype=torch.long, device=device)
                    size = min(common_tensor.numel(), self.config.k_max)
                    candidates[b, i, :size] = common_tensor[:size]
        return candidates

    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        B, L, _ = query_up.size()
        device = query_up.device
        start_time = time.time()
        query_groups = self.split_features_by_dim_groups(query_up)
        key_groups = self.split_features_by_dim_groups(key_up)
        cand_list = []
        for q_grp, k_grp in zip(query_groups, key_groups):
            cand_list.append(self._process_dimension_group_candidates(q_grp, k_grp, head_idx))
        if cand_list:
            merged = torch.cat(cand_list, dim=-1)
            merged, _ = torch.sort(merged, dim=-1)
            result = torch.unique_consecutive(merged, dim=-1)[:, :, :self.config.k_max]
        else:
            result = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=device)
        check_for_nan(result, "CandidateFinder output")
        check_for_inf(result, "CandidateFinder output")
        processing_time = time.time() - start_time
        if self.training and head_idx == 0:
            logger.debug(f"Candidate finding took {processing_time:.4f}s")
        return result

    def get_cache_stats(self):
        return {
            "wu_manber_hits": self.wu_manber_cache_hits,
            "wu_manber_misses": self.wu_manber_cache_misses,
            "trie_cache_size": len(self.trie_cache.cache)
        }

class AbsorptionProjection(nn.Module):
    """
    Improved AbsorptionProjection with optimized matrix operations.
    """
    def __init__(self, query_dim: int, key_dim: int, rank: int, dtype=torch.float32):
        super().__init__()
        std = 1.0 / math.sqrt(rank)
        self.u_q = nn.Parameter(torch.randn(query_dim, rank, dtype=dtype) * std)
        self.v_q = nn.Parameter(torch.randn(rank, key_dim, dtype=dtype) * std)
        self.u_k = nn.Parameter(torch.randn(key_dim, rank, dtype=dtype) * std)
        self.v_k = nn.Parameter(torch.randn(rank, key_dim, dtype=dtype) * std)
        self.register_buffer('W_absorb', None)
        self.needs_composition = True

    def _compute_absorption_matrix(self):
        W_UQ = torch.matmul(self.u_q, self.v_q)
        W_UK = torch.matmul(self.u_k, self.v_k)
        self.W_absorb = torch.matmul(W_UK.transpose(0, 1), W_UQ)
        self.needs_composition = False
        check_for_nan(self.W_absorb, "Absorption Matrix W_absorb")
        check_for_inf(self.W_absorb, "Absorption Matrix W_absorb")


    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.needs_composition or self.W_absorb is None:
            self._compute_absorption_matrix()
        Q_proj = torch.matmul(query, self.W_absorb.transpose(0, 1))
        check_for_nan(Q_proj, "Absorption Projection Q_proj")
        check_for_inf(Q_proj, "Absorption Projection Q_proj")
        return Q_proj, key

    def train(self, mode: bool = True):
        if mode and not self.training:
            self.needs_composition = True
        return super().train(mode)

class FastAttention(nn.Module):
    """
    Improved FastAttention with optimized computations and better parallelism.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        dtype = config.dtype

        self.query_down_proj = nn.Linear(config.d_model, config.d_query, dtype=dtype)
        self.key_value_down_proj = nn.Linear(config.d_model, config.d_key, dtype=dtype)

        self.absorption_projs = nn.ModuleList([
            AbsorptionProjection(config.d_query, config.d_key, config.rank, dtype=dtype)
            for _ in range(config.n_heads)
        ])
        self.value_up_projs = nn.ModuleList([
            LowRankLinear(config.d_key, config.d_model, config.rank, dtype=dtype)
            for _ in range(config.n_heads)
            ])

        if config.use_rff:
            self.rff_encoders = nn.ModuleList([
                RandomFourierFeatures(config.d_key, config.rff_dim, dtype=dtype)
                for _ in range(config.n_heads)
            ])
        self.candidate_finder = CandidateFinder(config)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model, dtype=dtype)
        self.dropout = nn.Dropout(config.dropout)
        self.forward_times = []

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        start_time = time.time()
        validate_tensor_dimensions(query, "query", query.dim())
        validate_tensor_dimensions(key, "key", key.dim())
        validate_tensor_dimensions(value, "value", value.dim())

        B, L, _ = query.size()
        device = query.device
        query_down = self.query_down_proj(query)
        key_down = self.key_value_down_proj(key)
        check_for_nan(query_down, "Query Down Projection")
        check_for_nan(key_down, "Key Down Projection")
        check_for_inf(query_down, "Query Down Projection")
        check_for_inf(key_down, "Key Down Projection")


        concat = torch.zeros(B, L, self.config.d_model * self.config.n_heads, device=device, dtype=query.dtype)

        for head_idx in range(self.config.n_heads):
            Q_proj, K_proj = self.absorption_projs[head_idx](query_down, key_down)
            check_for_nan(Q_proj, f"Head {head_idx} Absorption Projection Q_proj")
            check_for_nan(K_proj, f"Head {head_idx} Absorption Projection K_proj")
            check_for_inf(Q_proj, f"Head {head_idx} Absorption Projection Q_proj")
            check_for_inf(K_proj, f"Head {head_idx} Absorption Projection K_proj")


            candidates = self.candidate_finder(Q_proj, K_proj, head_idx)
            check_for_nan(candidates, f"Head {head_idx} CandidateFinder output")
            check_for_inf(candidates, f"Head {head_idx} CandidateFinder output")

            cand_mask = candidates != -1
            safe_candidates = candidates.masked_fill(~cand_mask, 0) # OUT-OF-PLACE
            num_candidates = candidates.size(-1)
            b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, L, num_candidates)
            candidate_keys = K_proj[b_idx, safe_candidates]
            check_for_nan(candidate_keys, f"Head {head_idx} Candidate Keys")
            check_for_inf(candidate_keys, f"Head {head_idx} Candidate Keys")

            q_exp = Q_proj.unsqueeze(2)
            if self.config.use_rff:
                rff_encoder = self.rff_encoders[head_idx]
                q_exp = rff_encoder(q_exp)
                candidate_keys = rff_encoder(candidate_keys)
                scale = optimized_sqrt(q_exp.size(-1))
            else:
                scale = optimized_sqrt(self.config.d_key)
            check_for_nan(q_exp, f"Head {head_idx} Q_exp after RFF")
            check_for_nan(candidate_keys, f"Head {head_idx} Candidate Keys after RFF")
            check_for_inf(q_exp, f"Head {head_idx} Q_exp after RFF")
            check_for_inf(candidate_keys, f"Head {head_idx} Candidate Keys after RFF")

            sim = torch.matmul(q_exp, candidate_keys.transpose(-2, -1)).squeeze(2) / scale
            check_for_nan(sim, f"Head {head_idx} Similarity Scores (sim)")
            check_for_inf(sim, f"Head {head_idx} Similarity Scores (sim)")


            sim = sim.masked_fill(~cand_mask, -1e9) # OUT-OF-PLACE
            if mask is not None:
                expanded_mask = mask.unsqueeze(1).expand_as(sim)
                sim = sim.masked_fill(~expanded_mask, -1e9) # OUT-OF-PLACE

            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)
            check_for_nan(sim, f"Head {head_idx} Sim before Softmax")
            check_for_nan(attn_weights, f"Head {head_idx} Attention Weights")
            check_for_inf(sim, f"Head {head_idx} Sim before Softmax")
            check_for_inf(attn_weights, f"Head {head_idx} Attention Weights")

            candidate_values = key_down[b_idx, safe_candidates]
            check_for_nan(candidate_values, f"Head {head_idx} Candidate Values")
            check_for_inf(candidate_values, f"Head {head_idx} Candidate Values")

            candidate_values = self.value_up_projs[head_idx](
                candidate_values.reshape(-1, self.config.d_key)
            ).reshape(B, L, num_candidates, self.config.d_model)
            check_for_nan(candidate_values, f"Head {head_idx} Candidate Values Up-Projected")
            check_for_inf(candidate_values, f"Head {head_idx} Candidate Values Up-Projected")


            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values, dim=2)
            check_for_nan(head_out, f"Head {head_idx} Output (head_out)")
            check_for_inf(head_out, f"Head {head_idx} Output (head_out)")
            concat_slice = slice(head_idx * self.config.d_model, (head_idx + 1) * self.config.d_model)
            concat[:, :, concat_slice] = head_out

        check_for_nan(concat, "Concat output before output_proj")
        check_for_inf(concat, "Concat output before output_proj")
        output = self.output_proj(concat)
        output = self.dropout(output)
        check_for_nan(output, "FastAttention final output")
        check_for_inf(output, "FastAttention final output")

        forward_time = time.time() - start_time
        self.forward_times.append(forward_time)
        if len(self.forward_times) > 100:
            self.forward_times = self.forward_times[-100:]
        if self.training and torch.rand(1).item() < 0.01:
            avg_time = sum(self.forward_times) / len(self.forward_times)
            logger.debug(f"FastAttention forward: {avg_time:.4f}s, batch={B}, seq_len={L}")
        return output

class FeedForwardNetwork(nn.Module):
    """
    Improved FeedForwardNetwork with GeLU and potential layer normalization.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        dtype = config.dtype
        self.linear1 = nn.Linear(config.d_model, config.intermediate_dim, dtype=dtype)
        self.linear2 = nn.Linear(config.intermediate_dim, config.d_model, dtype=dtype)
        self.dropout = nn.Dropout(config.dropout)
        self.use_layer_norm = True
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(config.intermediate_dim) # dtype is not needed (weight and bias are parameters specified above)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        check_for_nan(x, "FFN linear1 output")
        check_for_inf(x, "FFN linear1 output")
        x = F.gelu(x)
        if self.use_layer_norm:
            x = self.norm(x)
            check_for_nan(x, "FFN layer norm output")
            check_for_inf(x, "FFN layer norm output")
        x = self.dropout(x)
        x = self.linear2(x)
        check_for_nan(x, "FFN linear2 output")
        check_for_inf(x, "FFN linear2 output")
        x = self.dropout(x)
        return x

class FastAttentionEncoderLayer(nn.Module):
    """
    Improved FastAttentionEncoderLayer with Pre-Norm and residual connections.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.self_attn = FastAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.d_model)  # dtype is not needed (weight and bias are parameters)
        self.norm2 = nn.LayerNorm(config.d_model)  # dtype is not needed
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure src has the correct dtype
        src = src.to(self.norm1.weight.dtype)  # Match LayerNorm's parameters

        # Pre-Norm configuration
        x = self.norm1(src)
        check_for_nan(x, "Encoder Layer norm1 output")
        check_for_inf(x, "Encoder Layer norm1 output")
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        check_for_nan(attn_output, "Encoder Self Attention output")
        check_for_inf(attn_output, "Encoder Self Attention output")
        x = src + self.dropout(attn_output)
        x = self.norm2(x)
        check_for_nan(x, "Encoder Layer norm2 output")
        check_for_inf(x, "Encoder Layer norm2 output")
        ff_output = self.feed_forward(x)
        check_for_nan(ff_output, "Encoder FeedForward output")
        check_for_inf(ff_output, "Encoder FeedForward output")
        x = x + self.dropout(ff_output)
        return x

##############################################
# SPGPO Components - Refactored and Modularized
##############################################

class ResponseGenerator(nn.Module):
    def __init__(self, policy_network: FastAttentionEncoderLayer, k_responses: int, temperature: float = 1.0):
        super().__init__()
        self.policy_network = policy_network
        self.k_responses = k_responses
        self.temperature = temperature

    def generate_responses(self, prompt: torch.Tensor) -> torch.Tensor:
        B, L, d_model = prompt.size()  # Get d_model
        prompt_expanded = prompt.repeat_interleave(self.k_responses, dim=0) # (B*k_responses, L, d_model)
        check_for_nan(prompt_expanded, "ResponseGenerator prompt_expanded")
        check_for_inf(prompt_expanded, "ResponseGenerator prompt_expanded")

        # Ensure correct dtype for policy network input.
        prompt_expanded = prompt_expanded.to(self.policy_network.norm1.weight.dtype)

        logits = self.policy_network(prompt_expanded) / self.temperature  # (B*k_responses, L, vocab_size)
        check_for_nan(logits, "ResponseGenerator logits")
        check_for_inf(logits, "ResponseGenerator logits")

        probs = F.softmax(logits, dim=-1)
        check_for_nan(probs, "ResponseGenerator probs before Categorical")
        check_for_inf(probs, "ResponseGenerator probs before Categorical")
        dist = Categorical(probs)
        response_tokens = dist.sample()  # (B*k_responses, L)
        return response_tokens.view(B, self.k_responses, L)  # (B, k_responses, L)


def preference_oracle(response, response_group, prompt, reward_model):
    """
    Simplified preference oracle using the reward model.  Handles batches correctly.
    """
    B, K, L = response_group.shape
    rewards_responses = reward_model(prompt, response.squeeze(1))  # (B,)
    rewards_group = reward_model(prompt.repeat_interleave(K, dim=0), response_group.view(-1, L)).view(B, K) # (B, K)

    # Create preference matrix based on pairwise comparisons
    preferences = (rewards_responses.unsqueeze(1) > rewards_group).float()  # (B, K)
    return preferences


class PreferenceComputer:
    """
    Computes group preferences.
    """
    def __init__(self, reward_model: Callable):
        self.reward_model = reward_model

    def compute_group_preference(self, response: torch.Tensor, response_group: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:

        B, K, L = response_group.shape
         # (B, 1, L)
        preferences = preference_oracle(response, response_group, prompt, self.reward_model) # (B, K) Call the oracle
        return preferences

class AdvantageComputer:
    """
    Computes SPGPO advantage.
    """
    def __init__(self, preference_computer: PreferenceComputer, k_responses: int):
        self.preference_computer = preference_computer
        self.k_responses = k_responses

    def compute_spgpo_advantage(self, response: torch.Tensor, response_group: torch.Tensor, prompt: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        B, K, L = response_group.shape
        preferences = self.preference_computer.compute_group_preference(response, response_group, prompt)  #(B,K)
        mask = torch.ones((B, K), device=response.device)
        response_expanded = response.unsqueeze(1)
        equals = (response_expanded == response_group).all(dim=2)
        mask = mask.masked_fill(equals, 0) # OUT-OF-PLACE
        denom = torch.clamp(self.k_responses - equals.sum(dim=1, keepdim=True), min=1.0) # Avoid division by zero
        response_pref = (preferences * mask).sum(dim=1) / denom
        avg_group_pref = preferences.mean(dim=1)
        advantage = response_pref - avg_group_pref - baseline
        return advantage

class SPGPOEnvironment:
    """
    Environment for SPGPO training.
    """
    def __init__(self, policy_network: FastAttentionEncoderLayer, prompt_distribution: Callable,
                 reward_model: Callable, k_responses: int = 4, tokenizer: Callable = None,
                 clip_ratio: float = 0.2):
        self.prompt_distribution = prompt_distribution
        self.tokenizer = tokenizer
        self.clip_ratio = clip_ratio
        self.response_generator = ResponseGenerator(policy_network, k_responses)
        self.preference_computer = PreferenceComputer(reward_model)
        self.advantage_computer = AdvantageComputer(self.preference_computer, k_responses)
        self.policy_network = policy_network
        # Dummy tokenizer if none provided
        if self.tokenizer is None:
            self.tokenizer = lambda x: torch.randint(0, 100, (1, x))


    def step(self, prompts: torch.Tensor, old_log_probs_batch: torch.Tensor, baselines: torch.Tensor) -> torch.Tensor:
        B, L, d_model = prompts.shape # Get d_model
        all_responses = []
        all_advantages = []
        all_current_log_probs = []

        for i in range(B):
            prompt = prompts[i].unsqueeze(0) # (1, L, d_model)
            response_group = self.response_generator.generate_responses(prompt)  # (1, k_responses, L)
            response_group = response_group.squeeze(0) # (k_responses, L)
            response_group_expanded = response_group.unsqueeze(-1).repeat(1, 1, d_model)  # (k_responses, L) -> (k_responses, L, d_model)
            # Ensure correct dtype before passing to the policy network
            response_group_expanded = response_group_expanded.to(self.policy_network.norm1.weight.dtype)



            logits = self.policy_network(response_group_expanded)  # (k_responses, L, vocab_size)
            check_for_nan(logits, "Env Policy Network logits")
            check_for_inf(logits, "Env Policy Network logits")

            current_log_probs = F.log_softmax(logits, dim=-1) # (k_responses, L, vocab_size)
            check_for_nan(current_log_probs, "Env Log Probs")
            check_for_inf(current_log_probs, "Env Log Probs")

            for k in range(self.response_generator.k_responses):
                response = response_group[k]  # (L,)
                all_responses.append(response)
                advantage = self.advantage_computer.compute_spgpo_advantage(
                    response.unsqueeze(0), response_group.unsqueeze(0), prompt, baselines[i].unsqueeze(0)
                )
                all_advantages.append(advantage.squeeze(0))

                response_tokens = response.long()
                current_log_prob_response = current_log_probs[k]
                gathered_log_probs = torch.gather(current_log_prob_response, 1, response_tokens.unsqueeze(1)).squeeze(1)
                summed_log_prob = gathered_log_probs.sum()
                all_current_log_probs.append(summed_log_prob)

        all_responses = torch.stack(all_responses) # (B*k_responses, L)
        all_advantages = torch.stack(all_advantages) # (B*k_responses,)
        all_current_log_probs = torch.stack(all_current_log_probs)  # (B*k_responses,)

        log_diff = all_current_log_probs - old_log_probs_batch
        log_diff = torch.clamp(log_diff, -20, 20)  # Clip for stability
        ratios = torch.exp(log_diff)
        surr1 = ratios * all_advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * all_advantages
        loss = -torch.min(surr1, surr2).mean()
        return loss


class SPGPOAgent(nn.Module):
    """
    Agent class encapsulating the policy network.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.policy_network = FastAttentionEncoderLayer(config)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        # Ensure prompt has the correct dtype
        prompt = prompt.to(self.policy_network.norm1.weight.dtype)
        return self.policy_network(prompt)

def compute_baselines(prompts: torch.Tensor, env: SPGPOEnvironment) -> torch.Tensor:
    B, _, _ = prompts.shape
    baselines = []
    for i in range(B):
        prompt = prompts[i].unsqueeze(0)
        response_group = env.response_generator.generate_responses(prompt)
        response_group = response_group.squeeze(0) # (k_responses, L)
        preferences_batch = env.preference_computer.compute_group_preference(response_group, response_group.unsqueeze(0), prompt)
        avg_group_pref = preferences_batch.mean()  # Average preference for the group
        baselines.append(avg_group_pref)
    return torch.stack(baselines)

def spgpo_training_episode(agent: SPGPOAgent, env: SPGPOEnvironment, optimizer: optim.Optimizer,
                            scheduler: optim.lr_scheduler.LambdaLR, writer: SummaryWriter, episode: int,
                            batch_size: int = 32) -> float:

    total_loss = 0.0
    prompts = env.prompt_distribution(batch_size)
    B, L, d_model = prompts.shape # Get d_model

    # Compute baselines (using the current policy)
    baselines = compute_baselines(prompts, env).detach()

    # Calculate old log probabilities for the batch (outside the loop)
    old_log_probs_batch = []
    with torch.no_grad():  # No gradients needed for old log probs
        for i in range(batch_size):
            prompt = prompts[i].unsqueeze(0)
            response_group = env.response_generator.generate_responses(prompt)
            response_group = response_group.squeeze(0) # (k_responses, L)
            response_group_expanded = response_group.unsqueeze(-1).repeat(1, 1, d_model)  # (k_responses, L) -> (k_responses, L, d_model)
            # Ensure correct dtype before passing to the agent
            response_group_expanded = response_group_expanded.to(agent.policy_network.norm1.weight.dtype)

            logits = agent(response_group_expanded)  # (k_responses, L, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)

            for k in range(env.response_generator.k_responses):
                response = response_group[k]
                response_tokens = response.long()  # Ensure token indices are long
                log_prob_response = log_probs[k]
                gathered_log_probs = torch.gather(log_prob_response, 1, response_tokens.unsqueeze(1)).squeeze(1)
                summed_log_prob = gathered_log_probs.sum() # Sum log probabilities over the sequence
                old_log_probs_batch.append(summed_log_prob)

    old_log_probs_batch = torch.stack(old_log_probs_batch).detach() # Detach from the computation graph

    # Perform the policy update step
    loss = env.step(prompts, old_log_probs_batch, baselines)

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()
    scheduler.step()  # Update learning rate
    total_loss += loss.item()

    writer.add_scalar("Loss/Episode", loss.item(), episode)
    return total_loss


def train_spgpo(config: FastAttentionConfig, prompt_distribution: Callable, reward_model: Callable,
                 tokenizer: Callable, num_episodes: int = 100, batch_size: int = 32, k_responses: int = 4):

    dtype = config.dtype
    agent = SPGPOAgent(config)
    optimizer = optim.AdamW(agent.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    writer = SummaryWriter()

    env = SPGPOEnvironment(agent.policy_network, prompt_distribution, reward_model, k_responses, tokenizer, clip_ratio=0.2)

    print("Starting SPGPO Training...")
    for episode in range(num_episodes):
        episode_loss = spgpo_training_episode(agent, env, optimizer, scheduler, writer, episode, batch_size)
        print(f"Episode {episode+1}/{num_episodes}, Loss: {episode_loss:.4f}")

    writer.close()
    print("SPGPO Training Finished!")
    return agent

def main():
    # Example usage and training setup
    config = FastAttentionConfig(
        d_model=128,
        d_key=32,
        d_query=32,
        n_heads=4,
        rank=16,
        rff_dim=64,
        k_max=32,
        stride=8,
        lsh_buckets=64,
        lsh_bandwidth=2.0,
        lsh_key_dim=32,
        wu_manber_prefix_len=4,
        hyper_cuts_dim_groups=[16, 16],
        n_lsh_hashes=4,
        dropout=0.1,
        intermediate_dim=512,
        use_rff=True,
        dtype=torch.float32  # Explicitly set dtype
    )

    def simple_prompt_distribution(batch_size):
        seq_len = 64
        prompt_dim = config.d_model
        # Ensure the prompt tensor has the correct dtype.  THIS IS CRITICAL.
        return torch.randn(batch_size, seq_len, prompt_dim, dtype=config.dtype)

    def simple_reward_model(prompt, response):
        # Ensure consistent dtype for reward calculation
        return torch.randn(prompt.size(0), dtype=config.dtype)  # (B,)

    def simple_tokenizer(seq_len):
        # Consistent dtype
        return torch.randint(0, 100, (1, seq_len))

    trained_agent = train_spgpo(
        config=config,
        prompt_distribution=simple_prompt_distribution,
        reward_model=simple_reward_model,
        tokenizer=simple_tokenizer,
        num_episodes=50,  # Reduced for testing
        batch_size=16,
        k_responses=4
    )

    print("Trained Agent:", trained_agent)

if __name__ == "__main__":
    main()
