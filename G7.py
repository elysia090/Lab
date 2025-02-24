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
from torch.utils.checkpoint import checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

##############################################
# Low-level Helpers - Improved
##############################################

def optimized_sqrt(n: int) -> float:
    """Optimized square root for powers of 2."""
    if n <= 0:
        raise ValueError("Input must be positive")
    if n & (n - 1) == 0:
        k = n.bit_length() - 1
        return 2 ** (k / 2)
    return math.sqrt(n)

def fma(a: float, b: float, c: float) -> float:
    """Fused multiply-add with fallback to standard operations."""
    try:
        return math.fma(a, b, c)
    except AttributeError:
        return a * b + c

def validate_tensor_dimensions(tensor: torch.Tensor, name: str, expected_dims: int):
    """Validates tensor dimensions to provide clear error messages."""
    if tensor.dim() != expected_dims:
        raise ValueError(f"Expected {name} to have {expected_dims} dimensions, got {tensor.dim()}")

##############################################
# Fast Attention Components - Improved
##############################################

@dataclass
class FastAttentionConfig:
    """Configuration for FastAttention with validation."""
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
    """Improved LowRankLinear with better initialization and caching."""
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Better initialization for numerical stability
        std = 1.0 / math.sqrt(rank)
        self.u_weight = nn.Parameter(torch.randn(in_features, rank) * std)
        self.v_weight = nn.Parameter(torch.randn(rank, out_features) * std)

        # Cache the composed weight matrix for small ranks
        self.register_buffer('composed_weight', None)
        self.needs_composition = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cache the composed weight for efficiency if rank is small
        if self.rank <= min(self.in_features, self.out_features) // 4:
            if self.needs_composition or self.composed_weight is None:
                self.composed_weight = torch.matmul(self.u_weight, self.v_weight)
                self.needs_composition = False
            return torch.matmul(x, self.composed_weight)
        # Standard low-rank computation otherwise
        return torch.matmul(torch.matmul(x, self.u_weight), self.v_weight)

    def train(self, mode: bool = True):
        """Override train to reset composition cache when training."""
        if mode and not self.training:
            self.needs_composition = True
        return super().train(mode)

class RandomFourierFeatures(nn.Module):
    """Improved RandomFourierFeatures with normalized projections."""
    def __init__(self, input_dim: int, rff_dim: int):
        super().__init__()
        # Normalize the random projection for better numerical properties
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim) / math.sqrt(input_dim),
                                 requires_grad=False)
        self.bias = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi, requires_grad=False)
        self.scale = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_dimensions(x, "input", expected_dims=x.dim())
        # Handle reshaping internally for better usability
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])

        projection = x.matmul(self.omega) + self.bias
        result = torch.cos(projection) * self.scale

        # Restore original shape except for the last dimension
        if x.dim() > 2:
            result = result.reshape(*orig_shape[:-1], self.omega.size(1))

        return result

class LSHTable(nn.Module):
    """Improved LSHTable with better hashing and optional seed control."""
    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int, seed: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes

        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Use quasi-orthogonal vectors for initialization
        random_vectors = self._init_quasi_orthogonal_vectors(dim, n_hashes)
        self.register_buffer("random_vectors", random_vectors)

        # Counter for monitoring hash collisions
        self.register_buffer("collision_count", torch.zeros(1))

    def _init_quasi_orthogonal_vectors(self, dim: int, n_hashes: int) -> torch.Tensor:
        """Initialize quasi-orthogonal random vectors for better hashing."""
        vectors = torch.randn(dim, n_hashes)

        # Perform a simplified Gram-Schmidt process for small n_hashes
        if n_hashes <= dim and n_hashes <= 10:  # Limit to reasonable sizes
            for i in range(1, n_hashes):
                for j in range(i):
                    # Project and subtract
                    proj = torch.sum(vectors[:, i] * vectors[:, j]) / torch.sum(vectors[:, j] ** 2)
                    vectors[:, i] = vectors[:, i] - proj * vectors[:, j]

        # Normalize
        return vectors / torch.norm(vectors, dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        validate_tensor_dimensions(x, "input", x.dim())

        # Project and hash
        proj = x.matmul(self.random_vectors)
        hashed = torch.floor(proj / self.bandwidth) % self.n_buckets

        # Count hash collisions for monitoring (in training mode)
        if self.training:
            # Simple estimate of collisions using birthday paradox formula
            unique_hashes = torch.unique(hashed.reshape(-1, self.n_hashes), dim=0).shape[0]
            total_hashes = hashed.reshape(-1, self.n_hashes).shape[0]
            expected_unique = min(total_hashes, self.n_buckets ** self.n_hashes)
            if expected_unique > 0 and total_hashes > 0:
                self.collision_count[0] = 1.0 - (unique_hashes / expected_unique)

        return hashed

# Improve Trie with efficient storage
_TRIE_INDICES_KEY = '_indices'

class Trie(nn.Module):
    """Improved Trie with memory-efficient storage and faster lookups."""
    def __init__(self, stride: int):
        super().__init__()
        self.root_node: Dict = {}
        self.stride_len = stride
        # Track metrics for optimization
        self.max_depth = 0
        self.node_count = 0

    def insert(self, binary_vector: torch.Tensor, index: int) -> None:
        """Insert a binary vector into the trie."""
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

        # Store indices efficiently
        if _TRIE_INDICES_KEY not in current_node:
            current_node[_TRIE_INDICES_KEY] = []

        current_node[_TRIE_INDICES_KEY].append(index)
        self.max_depth = max(self.max_depth, depth)

    def search(self, binary_vector: torch.Tensor) -> List[int]:
        """Search for matching indices efficiently."""
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
        """Clear the trie to free memory."""
        self.root_node.clear()
        self.max_depth = 0
        self.node_count = 0

class LRUCache:
    """LRU Cache implementation."""
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        # アクセスされたアイテムを最新に移動
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # 既存のアイテムを更新し、最新に移動
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # 最も古いアイテムを削除
            self.cache.popitem(last=False)
        self.cache[key] = value


class TrieCache(nn.Module):
    """Cache of Tries for efficient reuse."""
    def __init__(self, stride: int, max_cache_size: int = 16):
        super().__init__()
        self.stride = stride
        self.max_cache_size = max_cache_size
        self.cache = LRUCache(max_cache_size)

    def get_trie(self, data_hash: int) -> Tuple[Trie, bool]:
        """Get a Trie for the given data hash, creating if needed."""
        cached_trie = self.cache.get(data_hash)
        if cached_trie is not None:
            return cached_trie, True

        # Create new Trie if not found
        trie = Trie(self.stride)
        self.cache.put(data_hash, trie)
        return trie, False

class CandidateFinder(nn.Module):
    """Improved CandidateFinder with caching and parallel processing."""
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        self.wu_manber_prefix_len = config.wu_manber_prefix_len
        self.hyper_cuts_dim_groups = config.hyper_cuts_dim_groups

        # Create LSH tables for each head
        self.lsh_tables = nn.ModuleList()
        for _ in range(config.n_heads):
            if config.hyper_cuts_dim_groups:
                head_tables = nn.ModuleList()
                for dim in config.hyper_cuts_dim_groups:
                    head_tables.append(
                        LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
                    )
                self.lsh_tables.append(head_tables)
            else:
                self.lsh_tables.append(nn.ModuleList([
                    LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
                ]))

        # Efficient trie caching
        self.trie_cache = TrieCache(config.stride)

        # Cache for Wu-Manber hash tables
        self.wu_manber_cache = LRUCache(capacity=1000) # LRU Cache
        self.wu_manber_cache_hits = 0
        self.wu_manber_cache_misses = 0

    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to binary values efficiently."""
        return (x > 0).float()

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        """Split features according to dimension groups."""
        if self.hyper_cuts_dim_groups is None:
            return [features]

        groups = []
        start = 0
        for group_dim in self.hyper_cuts_dim_groups:
            groups.append(features[:, :, start:start+group_dim])
            start += group_dim

        return groups

    def _build_wu_manber_hash_table(self, key_bin: torch.Tensor) -> Dict[tuple, List[int]]:
        """Build Wu-Manber hash table with improved caching."""
        # Generate a hash for the binary key - use a faster hashing method
        key_hash = hash(tuple(key_bin[:, :self.config.wu_manber_prefix_len].flatten().tolist()))

        # Check cache first
        cached_table = self.wu_manber_cache.get(key_hash) # LRU cache
        if cached_table is not None:
            self.wu_manber_cache_hits += 1
            return cached_table

        # Build new hash table more efficiently
        self.wu_manber_cache_misses += 1
        table: Dict[tuple, List[int]] = {}
        L = key_bin.size(0)

        # Use tensor operations for batch processing
        prefixes = [tuple(key_bin[i, :self.config.wu_manber_prefix_len].tolist()) for i in range(L)]
        for i, prefix in enumerate(prefixes):
            table.setdefault(prefix, []).append(i)

        # Use LRU cache
        self.wu_manber_cache.put(key_hash, table)

        return table

    def _wu_manber_search(self, query_bin: torch.Tensor, table: Dict[tuple, List[int]]) -> List[int]:
        """Search Wu-Manber hash table."""
        prefix = tuple(query_bin[:self.config.wu_manber_prefix_len].tolist())
        return table.get(prefix, [])

    def _get_wu_manber_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor) -> List[List[List[int]]]:
        """Get Wu-Manber candidates for a feature group."""
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
        """Get Trie candidates with batch processing optimization."""
        B, L, _ = key_grp.size()
        cand_lists = []

        # Pre-compute binary quantization once
        query_bin_all = self.binary_quantize(query_grp)

        for b in range(B):
            # Use a more efficient hash function
            key_data = key_grp[b].detach()
            # Only hash a subset of data for faster computation
            data_hash = hash(tuple(key_data[0, :50].tolist())) + hash(str(head_idx)) + b

            trie, cache_hit = self.trie_cache.get_trie(data_hash)

            if not cache_hit:
                # Build trie with vectorized operations where possible
                key_bin = self.binary_quantize(key_grp[b])
                for i in range(L):
                    trie.insert(key_bin[i], i)

            # Use pre-computed binary quantized queries
            query_bin = query_bin_all[b]

            # Process in chunks for better cache utilization
            batch_list = []
            chunk_size = 32  # Adjust based on hardware
            for chunk_start in range(0, L, chunk_size):
                chunk_end = min(chunk_start + chunk_size, L)
                chunk_results = [trie.search(query_bin[i]) for i in range(chunk_start, chunk_end)]
                batch_list.extend(chunk_results)

            cand_lists.append(batch_list)

        return cand_lists

    def merge_candidate_indices_groups(self, cand_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Merge candidate indices from different groups with optimized memory usage."""
        if not cand_tensors:
            return None # Or return torch.empty((B, L, 0), dtype=torch.long, device=device) if preferred

        # Allocate output tensor only once
        merged = torch.cat(cand_tensors, dim=-1)

        # Use in-place operations where possible
        merged, _ = torch.sort(merged, dim=-1) # Ensure dim=-1 for sorting along the candidate dimension
        # Use inplace unique to reduce memory overhead
        return torch.unique(merged, dim=-1)

    def _process_dimension_group_candidates(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> torch.Tensor:
        """Process candidates for dimension groups efficiently."""
        B, L, _ = query_grp.size()
        device = query_grp.device

        # Get candidates from both methods
        wu_cands = self._get_wu_manber_candidates_group(query_grp, key_grp)
        trie_cands = self._get_trie_candidates_group(query_grp, key_grp, head_idx)

        # Pre-allocate output tensor
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=device)

        # Process each batch and sequence element
        for b in range(B):
            for i in range(L):
                # Find common candidates (intersection)
                common = list(set(wu_cands[b][i]) & set(trie_cands[b][i]))

                if common:
                    # Convert to tensor and handle size constraints
                    common_tensor = torch.tensor(common, dtype=torch.long, device=device)
                    size = min(common_tensor.numel(), self.config.k_max)
                    candidates[b, i, :size] = common_tensor[:size]

        return candidates

    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        """Find candidates for attention efficiently."""
        B, L, _ = query_up.size()
        device = query_up.device

        start_time = time.time()

        # Split features by dimension groups
        query_groups = self.split_features_by_dim_groups(query_up)
        key_groups = self.split_features_by_dim_groups(key_up)

        # Process each dimension group
        cand_list = []
        for q_grp, k_grp in zip(query_groups, key_groups):
            cand_list.append(self._process_dimension_group_candidates(q_grp, k_grp, head_idx))

        # Merge candidates from all groups
        if cand_list:
            merged = self.merge_candidate_indices_groups(cand_list)
            result = merged[:, :, :self.config.k_max] if merged is not None else torch.full(
                (B, L, self.config.k_max), -1, dtype=torch.long, device=device
            )
        else:
            result = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=device)

        processing_time = time.time() - start_time
        if self.training and head_idx == 0 and processing_time > 0.1:  # Log only for first head and slow operations
            logger.debug(f"Candidate finding took {processing_time:.4f}s")

        return result

    def get_cache_stats(self):
        """Return cache statistics for monitoring."""
        return {
            "wu_manber_hits": self.wu_manber_cache_hits,
            "wu_manber_misses": self.wu_manber_cache_misses,
            "trie_cache_size": len(self.trie_cache.cache.cache) # Access inner cache of LRUCache
        }

class AbsorptionProjection(nn.Module):
    """Improved AbsorptionProjection with optimized matrix operations."""
    def __init__(self, query_dim: int, key_dim: int, rank: int):
        super().__init__()
        # Better initialization for stability
        std = 1.0 / math.sqrt(rank)
        self.u_q = nn.Parameter(torch.randn(query_dim, rank) * std)
        self.v_q = nn.Parameter(torch.randn(rank, key_dim) * std)
        self.u_k = nn.Parameter(torch.randn(key_dim, rank) * std)
        self.v_k = nn.Parameter(torch.randn(rank, key_dim) * std)

        # Cache for composed matrices
        self.register_buffer('W_absorb', None)
        self.needs_composition = True

    def _compute_absorption_matrix(self):
        """Compute and cache the absorption matrix."""
        W_UQ = torch.matmul(self.u_q, self.v_q)
        W_UK = torch.matmul(self.u_k, self.v_k)
        self.W_absorb = torch.matmul(W_UK.transpose(0, 1), W_UQ)
        self.needs_composition = False

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply absorption projection efficiently."""
        # Compute and cache absorption matrix if needed
        if self.needs_composition or self.W_absorb is None:
            self._compute_absorption_matrix()

        # Project query using cached matrix
        Q_proj = torch.matmul(query, self.W_absorb.transpose(0, 1))

        return Q_proj, key

    def train(self, mode: bool = True):
        """Override train to reset composition cache when training."""
        if mode and not self.training:
            self.needs_composition = True
        return super().train(mode)

class FastAttention(nn.Module):
    """Improved FastAttention with optimized computations and better parallelism."""
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config

        # Projections for queries and key-values
        self.query_down_proj = nn.Linear(config.d_model, config.d_query)
        self.key_value_down_proj = nn.Linear(config.d_model, config.d_key)

        # Absorption projections for each head
        self.absorption_projs = nn.ModuleList([
            AbsorptionProjection(config.d_query, config.d_key, config.rank)
            for _ in range(config.n_heads)
        ])

        # Value projections for each head
        self.value_up_projs = nn.ModuleList([
            LowRankLinear(config.d_key, config.d_model, config.rank)
            for _ in range(config.n_heads)
        ])

        # Random Fourier Features for each head
        if config.use_rff:
            self.rff_encoders = nn.ModuleList([
                RandomFourierFeatures(config.d_key, config.rff_dim)
                for _ in range(config.n_heads)
            ])

        # Single candidate finder with internal head-specific state
        self.candidate_finder = CandidateFinder(config)

        # Output projection
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Performance tracking
        self.forward_times = []

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optimized computations."""
        start_time = time.time()

        # Validate inputs
        validate_tensor_dimensions(query, "query", query.dim())
        validate_tensor_dimensions(key, "key", key.dim())
        validate_tensor_dimensions(value, "value", value.dim())

        B, L, _ = query.size()
        device = query.device

        # Project inputs
        query_down = self.query_down_proj(query)
        key_down = self.key_value_down_proj(key)

        # Process each attention head
        head_outputs = []
        for head_idx in range(self.config.n_heads):
            # Apply absorption projection
            Q_proj, K_proj = self.absorption_projs[head_idx](query_down, key_down)

            # Find attention candidates
            candidates = self.candidate_finder(Q_proj, K_proj, head_idx)

            # Create mask for valid candidates
            cand_mask = candidates != -1

            # Handle invalid indices safely
            safe_candidates = candidates.clone()
            safe_candidates[~cand_mask] = 0

            # Get dimensions
            num_candidates = candidates.size(-1)

            # Create batch indices for gathering
            b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, L, num_candidates)

            # Gather candidate keys
            candidate_keys = K_proj[b_idx, safe_candidates]

            # Apply Random Fourier Features if enabled
            q_exp = Q_proj.unsqueeze(2)
            if self.config.use_rff:
                rff_encoder = self.rff_encoders[head_idx]

                # RFFをクエリに一度だけ適用（reshapeの操作を最小限に）
                q_rff = rff_encoder(Q_proj.view(-1, self.config.d_key))
                q_exp = q_rff.view(B, L, self.config.rff_dim).unsqueeze(2)

                # 候補キーを集める前にK_projにRFFを適用（より効率的な場合がある）
                K_rff = rff_encoder(K_proj.reshape(-1, self.config.d_key)).reshape(B, L, self.config.rff_dim)

                # 安全な候補のみを集める
                candidate_keys = K_rff[b_idx, safe_candidates]
                scale = 1.0 / optimized_sqrt(self.config.rff_dim)  # 数値的安定性のために逆数を事前計算
            else:
                q_exp = Q_proj.unsqueeze(2)
                candidate_keys = K_proj[b_idx, safe_candidates]
                scale = 1.0 / optimized_sqrt(self.config.d_key)  # 数値的安定性のために逆数を事前計算

            # スケールを乗算することで、除算よりも速い乗算を使用
            sim = torch.matmul(q_exp, candidate_keys.transpose(-2, -1)).squeeze(2) * scale

            # Mask invalid candidates
            sim = sim.masked_fill(~cand_mask, float('-inf'))

            # Apply optional attention mask
            if mask is not None:
                # Expand mask for broadcasting
                expanded_mask = mask.unsqueeze(1).expand_as(sim)
                sim = sim.masked_fill(~expanded_mask, float('-inf'))

            # Compute attention weights
            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Gather candidate values
            candidate_values = key_down[b_idx, safe_candidates]

            # Project values to output dimension
            candidate_values = self.value_up_projs[head_idx](
                candidate_values.reshape(-1, self.config.d_key)
            ).reshape(B, L, num_candidates, self.config.d_model)

            # Apply attention weights to values
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values, dim=2)
            head_outputs.append(head_out)

        # Concatenate head outputs
        concat = torch.cat(head_outputs, dim=-1)

        # Apply output projection and dropout
        output = self.output_proj(concat)
        output = self.dropout(output)

        # Track performance
        forward_time = time.time() - start_time
        self.forward_times.append(forward_time)
        if len(self.forward_times) > 100:  # Keep only recent times
            self.forward_times.pop(0)

        if self.training and torch.rand(1).item() < 0.01:  # Log occasionally during training
            avg_time = sum(self.forward_times) / len(self.forward_times)
            logger.debug(f"FastAttention forward: {avg_time:.4f}s, batch={B}, seq_len={L}")

        return output

class FeedForwardNetwork(nn.Module):
    """Improved FeedForwardNetwork with GeLU and potential layer normalization."""
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.intermediate_dim)
        self.linear2 = nn.Linear(config.intermediate_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Optional layer norm for more stable gradients
        self.use_layer_norm = False
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(config.intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with improved activations and optional normalization."""
        x = self.linear1(x)
        x = F.gelu(x)  # Use GeLU for smoother gradients
        if self.use_layer_norm:
            x = self.norm(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class FastAttentionEncoderLayer(nn.Module):
    """Improved FastAttentionEncoderLayer with Pre-Norm, residual connections, and gradient checkpointing."""
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.self_attn = FastAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with Pre-Norm, residual connections and gradient checkpointing."""
        # メモリ効率化のための勾配チェックポイント
        if self.training and src.requires_grad:
            return checkpoint(self._forward_impl, src, src_mask)
        else:
            return self._forward_impl(src, src_mask)

    def _forward_impl(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Implementation of forward pass."""
        # Pre-Norm configuration
        x = self.norm1(src)
        x = src + self.dropout(self.self_attn(x, x, x, mask=src_mask))  # Residual connection
        x = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x))  # Residual connection
        return x

##############################################
# SPGPO Components - Refactored and Modularized
##############################################

def preference_oracle(y1_batch: torch.Tensor, y2_batch: torch.Tensor, x_batch: torch.Tensor, reward_model: Callable) -> torch.Tensor:
    """Batched preference oracle using a reward model."""
    with torch.no_grad():
        score1 = reward_model(x_batch, y1_batch)
        score2 = reward_model(x_batch, y2_batch)
    return torch.sigmoid(score1 - score2)

class ResponseGenerator:
    """Generates responses from the policy network."""
    def __init__(self, policy_network: FastAttentionEncoderLayer, k_responses: int, temperature: float = 1.0):
        self.policy_network = policy_network
        self.k_responses = k_responses
        self.temperature = temperature  # Add temperature

    def generate_responses(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generates multiple responses (batched)."""
        B, L, _ = prompt.size()
        prompt_expanded = prompt.repeat_interleave(self.k_responses, dim=0)
        logits = self.policy_network(prompt_expanded) / self.temperature # Apply temperature

        # Use Categorical distribution for sampling.
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        response_tokens = dist.sample()

        return response_tokens.view(B, self.k_responses, -1) # Ensure correct shape (B, k, L)


class PreferenceComputer:
    """Computes group preferences."""
    def __init__(self, reward_model: Callable):
        self.reward_model = reward_model

    def compute_group_preference(self, response: torch.Tensor, response_group: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """Computes batched group preference P(y ≻ G_x\{y}|x)."""
        B, K, L = response_group.shape
        response_expanded = response.unsqueeze(1)  # (B, 1, L)
        response_group_expanded = response_group.unsqueeze(0) # (1, K, L)
        prompt_expanded = prompt.unsqueeze(1) # (B, 1, prompt_dim)
        preferences = preference_oracle(response_expanded, response_group_expanded, prompt_expanded, self.reward_model)
        return preferences

class AdvantageComputer:
    """Computes SPGPO advantage."""
    def __init__(self, preference_computer: PreferenceComputer, k_responses: int):
        self.preference_computer = preference_computer
        self.k_responses = k_responses

    def compute_spgpo_advantage(self, response: torch.Tensor, response_group: torch.Tensor, prompt: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        """Computes batched SPGPO advantage A_SPGPO(y|x) with baseline."""
        B, K, L = response_group.shape
        preferences = self.preference_computer.compute_group_preference(response, response_group, prompt)
        mask = torch.ones((B, K), device=response.device)
        # Find response and create a mask. Much faster than torch.equal in a loop
        for b in range(B):
            response_b = response[b].unsqueeze(0)  # (1, L)
            expanded_group = response_group[b] # (K, L)
            equals = (response_b.unsqueeze(1) == expanded_group.unsqueeze(0)).all(-1).any(0) # Check along length.
            mask[b, equals] = 0

        response_pref = (preferences * mask).sum(dim=1) / (self.k_responses - 1 + 1e-8)
        avg_group_pref = preferences.mean(dim=1)
        advantage = response_pref - avg_group_pref - baseline
        return advantage

class SPGPOEnvironment:
    """Environment for SPGPO training, orchestrating response generation, preference, and advantage computation."""
    def __init__(self, policy_network: FastAttentionEncoderLayer, prompt_distribution: Callable,
                 reward_model: Callable, k_responses: int = 4, tokenizer: Callable = None,
                 clip_ratio: float = 0.2, logging: bool = False):
        """Initializes the SPGPO environment with modular components."""
        self.prompt_distribution = prompt_distribution
        self.tokenizer = tokenizer
        self.clip_ratio = clip_ratio
        self.logging = logging

        # Modular components
        self.response_generator = ResponseGenerator(policy_network, k_responses)
        self.preference_computer = PreferenceComputer(reward_model)
        self.advantage_computer = AdvantageComputer(self.preference_computer, k_responses)
        self.policy_network = policy_network # Directly use policy network for log prob calculation


        if self.tokenizer is None:
            self.tokenizer = lambda x: torch.randint(0, 100, (1, x))


    def step(self, prompts: torch.Tensor, old_log_probs_batch: torch.Tensor, baselines: torch.Tensor) -> torch.Tensor:
        """Performs a batched step in the SPGPO environment using modular components."""
        start_time = time.time()
        B, _, _ = prompts.shape
        all_responses = []
        all_advantages = []
        all_current_log_probs = []

        for i in range(B):
            prompt = prompts[i].unsqueeze(0)
            response_group = self.response_generator.generate_responses(prompt)
            response_group = response_group.squeeze(0)

            logits = self.policy_network(response_group)
            current_log_probs = F.log_softmax(logits, dim=-1)

            for k in range(self.response_generator.k_responses): # Use k_responses from generator
                response = response_group[k]
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

        all_responses = torch.stack(all_responses)
        all_advantages = torch.stack(all_advantages)
        all_current_log_probs = torch.stack(all_current_log_probs)

        # Add clipping to prevent extreme values
        log_diff = all_current_log_probs - old_log_probs_batch
        log_diff = torch.clamp(log_diff, -20, 20)  # Prevent extreme values
        ratios = torch.exp(log_diff)

        surr1 = ratios * all_advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * all_advantages
        loss = -torch.min(surr1, surr2).mean()

        # パフォーマンスのモニタリング追加
        step_time = time.time() - start_time
        if self.logging and B > 1:  # バッチ処理の場合のみログを記録
            logger.debug(f"SPGPO step for batch size {B}: {step_time:.4f}s "
                         f"({step_time/B:.4f}s per prompt)")
        return loss


class SPGPOAgent(nn.Module):
    """Agent class encapsulating the policy network."""
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.policy_network = FastAttentionEncoderLayer(config)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        return self.policy_network(prompt)

def compute_baselines(prompts: torch.Tensor, env: SPGPOEnvironment) -> torch.Tensor:
    """Computes baselines for each prompt using average group preference."""
    B, _, _ = prompts.shape
    baselines = []
    for i in range(B):
        prompt = prompts[i].unsqueeze(0)
        response_group = env.response_generator.generate_responses(prompt) # Use response generator
        response_group = response_group.squeeze(0)
        preferences_batch = env.preference_computer.compute_group_preference(response_group, response_group.unsqueeze(0), prompt) # Use preference computer

        avg_group_pref = preferences_batch.mean()
        baselines.append(avg_group_pref)
    return torch.stack(baselines)


def spgpo_training_episode(agent: SPGPOAgent, env: SPGPOEnvironment, optimizer: optim.Optimizer,
                            scheduler: optim.lr_scheduler.LambdaLR, writer: SummaryWriter, episode: int,
                            batch_size: int = 32) -> float:
    """Runs a single SPGPO training episode with batched prompts and PPO-style loss."""
    total_loss = 0.0
    prompts = env.prompt_distribution(batch_size)

    baselines = compute_baselines(prompts, env).detach()

    old_log_probs_batch = []
    with torch.no_grad():
        for i in range(batch_size):
            prompt = prompts[i].unsqueeze(0)
            response_group = env.response_generator.generate_responses(prompt) # Use response generator
            response_group = response_group.squeeze(0)

            logits = agent(response_group)
            log_probs = F.log_softmax(logits, dim=-1)

            for k in range(env.response_generator.k_responses): # Use k_responses from generator
                response = response_group[k]
                response_tokens = response.long()
                log_prob_response = log_probs[k]
                gathered_log_probs = torch.gather(log_prob_response, 1, response_tokens.unsqueeze(1)).squeeze(1)
                summed_log_prob = gathered_log_probs.sum()
                old_log_probs_batch.append(summed_log_prob)

    old_log_probs_batch = torch.stack(old_log_probs_batch).detach()


    loss = env.step(prompts, old_log_probs_batch, baselines)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()

    writer.add_scalar("Loss/Episode", loss.item(), episode)
    return total_loss


def train_spgpo(config: FastAttentionConfig, prompt_distribution: Callable, reward_model: Callable,
                 tokenizer: Callable, num_episodes: int = 100, batch_size: int = 32, k_responses: int = 4):
    """Main training loop for SPGPO with batched training and TensorBoard logging."""
    agent = SPGPOAgent(config)
    optimizer = optim.AdamW(agent.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    writer = SummaryWriter()

    env = SPGPOEnvironment(agent.policy_network, prompt_distribution, reward_model, k_responses, tokenizer, clip_ratio=0.2, logging=True) # Pass policy_network and enable logging

    print("Starting SPGPO Training...")
    for episode in range(num_episodes):
        episode_loss = spgpo_training_episode(agent, env, optimizer, scheduler, writer, episode, batch_size)
        print(f"Episode {episode+1}/{num_episodes}, Loss: {episode_loss:.4f}")

    writer.close()
    print("SPGPO Training Finished!")
    return agent

def main():
    # Example usage and training setup (adjust as needed)
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
        hyper_cuts_dim_groups=[16, 16], # Example dimension groups
        n_lsh_hashes=4,
        dropout=0.1,
        intermediate_dim=512,
        use_rff=True
    )

    def simple_prompt_distribution(batch_size):
        # Dummy prompt distribution for example
        seq_len = 64
        prompt_dim = config.d_model
        return torch.randn(batch_size, seq_len, prompt_dim)

    def simple_reward_model(prompt, response):
        # Dummy reward model for example
        return torch.randn(prompt.size(0))

    def simple_tokenizer(seq_len):
        # Dummy tokenizer
        return torch.randint(0, 100, (1, seq_len))

    # Train the SPGPO agent
    trained_agent = train_spgpo(
        config=config,
        prompt_distribution=simple_prompt_distribution,
        reward_model=simple_reward_model,
        tokenizer=simple_tokenizer,
        num_episodes=50,  # Reduced for example
        batch_size=16,    # Reduced for example
        k_responses=4
    )

    print("Trained Agent:", trained_agent)
    # You can now use trained_agent for inference or further training

if __name__ == "__main__":
    main()
