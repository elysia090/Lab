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


##############################################
# Fast Attention Components - Improved
##############################################

@dataclass
class FastAttentionConfig:
    """
    Configuration for FastAttention with validation.

    Attributes:
        d_model (int): Dimension of the model.
        d_key (int): Dimension of key vectors.
        d_query (int): Dimension of query vectors.
        n_heads (int): Number of attention heads.
        rank (int): Rank for low-rank approximations.
        rff_dim (int): Dimension of Random Fourier Features.
        k_max (int): Maximum number of candidates to consider.
        stride (int): Stride for Trie data structure.
        lsh_buckets (int): Number of buckets for LSH.
        lsh_bandwidth (float): Bandwidth for LSH hashing.
        lsh_key_dim (int): Dimension of keys used in LSH.
        wu_manber_prefix_len (int): Prefix length for Wu-Manber algorithm.
        hyper_cuts_dim_groups (Optional[List[int]]): Dimension groups for HyperCuts LSH.
        n_lsh_hashes (int): Number of LSH hashes.
        dropout (float): Dropout probability.
        intermediate_dim (int): Intermediate dimension for FeedForwardNetwork.
        use_rff (bool): Whether to use Random Fourier Features.
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

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        rank (int): Rank of the low-rank approximation.

    Input:
        x (torch.Tensor): Input tensor of shape (..., in_features).

    Output:
        torch.Tensor: Output tensor of shape (..., out_features).
    """
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
        """
        Forward pass of LowRankLinear.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
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
    """
    Improved RandomFourierFeatures with normalized projections and JIT compatibility.

    Args:
        input_dim (int): Dimension of the input features.
        rff_dim (int): Dimension of the Random Fourier Features.

    Input:
        x (torch.Tensor): Input tensor of shape (..., input_dim).

    Output:
        torch.Tensor: Output tensor of shape (..., rff_dim).
    """
    def __init__(self, input_dim: int, rff_dim: int):
        super().__init__()
        # Normalize the random projection for better numerical properties
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim) / math.sqrt(input_dim),
                                 requires_grad=False)
        self.bias = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi, requires_grad=False)
        self.scale = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RandomFourierFeatures.

        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (..., rff_dim).
        """
        # Store original shape for reshaping
        orig_shape = x.shape

        # Reshape for batch matmul if needed
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1]) # Reshape to (N, input_dim) for matrix multiplication

        # Use efficient matrix operations
        projection = torch.addmm(self.bias, x, self.omega) # (N, rff_dim) = bias + x @ omega
        result = torch.cos(projection) * self.scale # (N, rff_dim)
        check_for_nan(result, "RFF output") # NaN check

        # Restore original shape except for the last dimension
        if len(orig_shape) > 2:
            new_shape = list(orig_shape[:-1]) + [self.omega.size(1)] # Original shape with last dim replaced by rff_dim
            result = result.view(*new_shape) # Reshape back to original shape (..., rff_dim)

        return result

class LSHTable(nn.Module):
    """
    Improved LSHTable with better hashing and optional seed control.

    Args:
        dim (int): Dimension of the input vectors.
        n_buckets (int): Number of LSH buckets.
        bandwidth (float): Bandwidth for LSH hashing.
        n_hashes (int): Number of LSH hashes.
        seed (Optional[int]): Random seed for reproducibility.

    Input:
        x (torch.Tensor): Input tensor of shape (..., dim).

    Output:
        torch.Tensor: Hashed tensor of shape (..., n_hashes) containing bucket indices.
    """
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
        random_vectors = self._init_quasi_orthogonal_vectors(dim, n_hashes) # (dim, n_hashes)
        self.register_buffer("random_vectors", random_vectors)

        # Counter for monitoring hash collisions
        self.register_buffer("collision_count", torch.zeros(1))

    def _init_quasi_orthogonal_vectors(self, dim: int, n_hashes: int) -> torch.Tensor:
        """Initialize quasi-orthogonal random vectors for better hashing."""
        vectors = torch.randn(dim, n_hashes) # (dim, n_hashes)

        # Perform a simplified Gram-Schmidt process for small n_hashes
        if n_hashes <= dim and n_hashes <= 10:  # Limit to reasonable sizes
            for i in range(1, n_hashes):
                for j in range(i):
                    # Project and subtract
                    proj = torch.sum(vectors[:, i] * vectors[:, j]) / torch.sum(vectors[:, j] ** 2)
                    vectors[:, i] = vectors[:, i] - proj * vectors[:, j]

        # Normalize
        return vectors / torch.norm(vectors, dim=0, keepdim=True) # (dim, n_hashes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LSHTable.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Hashed tensor of shape (..., n_hashes) containing bucket indices.
        """
        validate_tensor_dimensions(x, "input", x.dim())

        # Project and hash
        proj = x.matmul(self.random_vectors) # (..., n_hashes) = (..., dim) @ (dim, n_hashes)
        hashed = torch.floor(proj / self.bandwidth) % self.n_buckets # (..., n_hashes)
        check_for_nan(hashed, "LSH hashed output") # NaN check

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
    """
    Improved Trie with memory-efficient storage and faster lookups.

    Args:
        stride (int): Stride length for processing binary vectors.
    """
    def __init__(self, stride: int):
        super().__init__()
        self.root_node: Dict = {}
        self.stride_len = stride
        # Track metrics for optimization
        self.max_depth = 0
        self.node_count = 0

    def insert(self, binary_vector: torch.Tensor, index: int) -> None:
        """
        Insert a binary vector into the trie.

        Args:
            binary_vector (torch.Tensor): 1D binary vector to insert.
            index (int): Index associated with the binary vector.

        Raises:
            ValueError: If binary_vector is not 1D.
        """
        if binary_vector.dim() != 1:
            raise ValueError(f"Expected 1D binary_vector, got {binary_vector.dim()}D")

        current_node = self.root_node
        depth = 0

        for i in range(0, len(binary_vector), self.stride_len):
            depth += 1
            end_idx = min(i + self.stride_len, len(binary_vector))
            prefix = tuple(binary_vector[i:end_idx].tolist()) # Convert prefix to tuple for dict key

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
        """
        Search for matching indices efficiently.

        Args:
            binary_vector (torch.Tensor): 1D binary vector to search for.

        Returns:
            List[int]: List of indices associated with the binary vector.

        Raises:
            ValueError: If binary_vector is not 1D.
        """
        if binary_vector.dim() != 1:
            raise ValueError(f"Expected 1D binary_vector, got {binary_vector.dim()}D")

        current_node = self.root_node

        for i in range(0, len(binary_vector), self.stride_len):
            end_idx = min(i + self.stride_len, len(binary_vector))
            prefix = tuple(binary_vector[i:end_idx].tolist()) # Convert prefix to tuple for dict key

            if prefix not in current_node:
                return []

            current_node = current_node[prefix]

        return current_node.get(_TRIE_INDICES_KEY, [])

    def clear(self) -> None:
        """Clear the trie to free memory."""
        self.root_node.clear()
        self.max_depth = 0
        self.node_count = 0

class TrieCache(nn.Module):
    """
    Cache of Tries for efficient reuse.

    Args:
        stride (int): Stride length for Tries in the cache.
        max_cache_size (int): Maximum number of Tries to cache.
    """
    def __init__(self, stride: int, max_cache_size: int = 16):
        super().__init__()
        self.stride = stride
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()  # Use OrderedDict for FIFO behavior

    def get_trie(self, data_hash: int) -> Tuple[Trie, bool]:
        """
        Get a Trie for the given data hash, creating if needed.

        Args:
            data_hash (int): Hash value to identify the Trie.

        Returns:
            Tuple[Trie, bool]: Tuple containing the Trie and a boolean indicating cache hit.
        """
        if data_hash in self.cache:
            trie = self.cache.pop(data_hash) # Pop and re-insert to move to end (FIFO-like for eviction)
            self.cache[data_hash] = trie
            return trie, True

        # Create new Trie if not found
        trie = Trie(self.stride)

        # Manage cache size - FIFO eviction using OrderedDict
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False) # Remove oldest item (FIFO)

        self.cache[data_hash] = trie
        return trie, False

class CandidateFinder(nn.Module):
    """
    Improved CandidateFinder with caching and parallel processing.

    Args:
        config (FastAttentionConfig): Configuration for FastAttention.
    """
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
        self.wu_manber_cache = OrderedDict() # Use OrderedDict for FIFO cache
        self.wu_manber_cache_hits = 0
        self.wu_manber_cache_misses = 0

    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize tensor to binary values efficiently.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Binary quantized tensor (0 or 1).
        """
        return (x > 0).float()

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Split features according to dimension groups.

        Args:
            features (torch.Tensor): Input features tensor of shape (B, L, lsh_key_dim).

        Returns:
            List[torch.Tensor]: List of feature tensors split by dimension groups.
        """
        if self.hyper_cuts_dim_groups is None:
            return [features]

        groups = []
        start = 0
        for group_dim in self.hyper_cuts_dim_groups:
            groups.append(features[:, :, start:start+group_dim]) # (B, L, group_dim)
            start += group_dim

        return groups

    def _build_wu_manber_hash_table(self, key_bin: torch.Tensor) -> Dict[tuple, List[int]]:
        """
        Build Wu-Manber hash table with improved caching.

        Args:
            key_bin (torch.Tensor): Binary quantized key tensor of shape (L, lsh_key_dim).

        Returns:
            Dict[tuple, List[int]]: Wu-Manber hash table.
        """
        # Generate a hash for the binary key - use a faster hashing method
        key_hash = hash(tuple(key_bin[:, :self.config.wu_manber_prefix_len].flatten().tolist()))

        # Check cache first
        if key_hash in self.wu_manber_cache:
            self.wu_manber_cache_hits += 1
            table = self.wu_manber_cache.pop(key_hash) # Pop and re-insert for FIFO-like eviction
            self.wu_manber_cache[key_hash] = table
            return table

        # Build new hash table more efficiently
        self.wu_manber_cache_misses += 1
        table: Dict[tuple, List[int]] = {}
        L = key_bin.size(0)

        # Use tensor operations for batch processing
        prefixes = [tuple(key_bin[i, :self.config.wu_manber_prefix_len].tolist()) for i in range(L)]
        for i, prefix in enumerate(prefixes):
            table.setdefault(prefix, []).append(i)

        # Use LRU cache with ordered dict instead of arbitrary removal
        self.wu_manber_cache[key_hash] = table

        # More efficient cache management with max size
        if len(self.wu_manber_cache) > 1000:
            # Use FIFO approach for more predictable cache behavior
            self.wu_manber_cache.popitem(last=False) # Remove oldest item (FIFO)

        return table

    def _wu_manber_search(self, query_bin: torch.Tensor, table: Dict[tuple, List[int]]) -> List[int]:
        """
        Search Wu-Manber hash table.

        Args:
            query_bin (torch.Tensor): Binary quantized query tensor of shape (lsh_key_dim,).
            table (Dict[tuple, List[int]]): Wu-Manber hash table.

        Returns:
            List[int]: List of candidate indices from Wu-Manber search.
        """
        prefix = tuple(query_bin[:self.config.wu_manber_prefix_len].tolist())
        return table.get(prefix, [])

    def _get_wu_manber_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor) -> List[List[List[int]]]:
        """
        Get Wu-Manber candidates for a feature group.

        Args:
            query_grp (torch.Tensor): Query feature group of shape (B, L, group_dim).
            key_grp (torch.Tensor): Key feature group of shape (B, L, group_dim).

        Returns:
            List[List[List[int]]]: List of candidate indices for each batch and sequence element.
        """
        B, L, _ = key_grp.size()
        key_bin = self.binary_quantize(key_grp) # (B, L, group_dim) -> (B, L, group_dim) (binary)
        query_bin = self.binary_quantize(query_grp) # (B, L, group_dim) -> (B, L, group_dim) (binary)
        cand_lists = []

        for b in range(B):
            table = self._build_wu_manber_hash_table(key_bin[b]) # Build table for each batch (L, group_dim)
            batch_list = [self._wu_manber_search(query_bin[b, i], table) for i in range(L)] # Search for each sequence element
            cand_lists.append(batch_list)

        return cand_lists

    def _get_trie_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> List[List[List[int]]]:
        """
        Get Trie candidates with batch processing optimization.

        Args:
            query_grp (torch.Tensor): Query feature group of shape (B, L, group_dim).
            key_grp (torch.Tensor): Key feature group of shape (B, L, group_dim).
            head_idx (int): Index of the attention head.

        Returns:
            List[List[List[int]]]: List of candidate indices for each batch and sequence element from Trie search.
        """
        B, L, _ = key_grp.size()
        cand_lists = []

        # Pre-compute binary quantization once
        query_bin_all = self.binary_quantize(query_grp) # (B, L, group_dim) -> (B, L, group_dim) (binary)

        for b in range(B):
            # Use a more efficient hash function
            key_data = key_grp[b].detach()
            # Only hash a subset of data for faster computation
            data_hash = hash(tuple(key_data[0, :50].tolist())) + hash(str(head_idx)) + b

            trie, cache_hit = self.trie_cache.get_trie(data_hash)

            if not cache_hit:
                # Build trie with vectorized operations where possible
                key_bin = self.binary_quantize(key_grp[b]) # (L, group_dim) -> (L, group_dim) (binary)
                for i in range(L):
                    trie.insert(key_bin[i], i) # Insert binary key and index into Trie

            # Use pre-computed binary quantized queries
            query_bin = query_bin_all[b] # (L, group_dim) (binary)

            # Process in chunks for better cache utilization
            batch_list = []
            chunk_size = 32  # Adjust based on hardware
            for chunk_start in range(0, L, chunk_size):
                chunk_end = min(chunk_start + chunk_size, L)
                chunk_results = [trie.search(query_bin[i]) for i in range(chunk_start, chunk_end)] # Search Trie for each query
                batch_list.extend(chunk_results)

            cand_lists.append(batch_list)

        return cand_lists

    def merge_candidate_indices_groups(self, cand_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge candidate indices from different groups with optimized memory usage.

        Args:
            cand_tensors (List[torch.Tensor]): List of candidate index tensors, each of shape (B, L, *).

        Returns:
            torch.Tensor: Merged and unique candidate indices tensor of shape (B, L, num_candidates).
        """
        if not cand_tensors:
            return None # Or return torch.empty((B, L, 0), dtype=torch.long, device=device) if preferred

        # Allocate output tensor only once
        merged = torch.cat(cand_tensors, dim=-1) # (B, L, total_candidates)

        # Use in-place operations where possible
        merged, _ = torch.sort(merged, dim=-1) # Ensure dim=-1 for sorting along the candidate dimension
        # Use inplace unique to reduce memory overhead
        return torch.unique(merged, dim=-1) # (B, L, unique_candidates)

    def _process_dimension_group_candidates(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        Process candidates for dimension groups efficiently.

        Args:
            query_grp (torch.Tensor): Query feature group of shape (B, L, group_dim).
            key_grp (torch.Tensor): Key feature group of shape (B, L, group_dim).
            head_idx (int): Index of the attention head.

        Returns:
            torch.Tensor: Processed candidate indices tensor of shape (B, L, k_max).
        """
        B, L, _ = query_grp.size()
        device = query_grp.device

        # Get candidates from both methods
        wu_cands = self._get_wu_manber_candidates_group(query_grp, key_grp) # List[List[List[int]]]
        trie_cands = self._get_trie_candidates_group(query_grp, key_grp, head_idx) # List[List[List[int]]]

        # Pre-allocate output tensor
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=device) # (B, L, k_max)

        # Process each batch and sequence element
        for b in range(B):
            for i in range(L):
                # Find common candidates (intersection)
                common = list(set(wu_cands[b][i]) & set(trie_cands[b][i]))

                if common:
                    # Convert to tensor and handle size constraints
                    common_tensor = torch.tensor(common, dtype=torch.long, device=device) # (len(common),)
                    size = min(common_tensor.numel(), self.config.k_max)
                    candidates[b, i, :size] = common_tensor[:size] # Fill candidates tensor

        return candidates

    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        Find candidates for attention efficiently.

        Args:
            query_up (torch.Tensor): Query tensor after up-projection of shape (B, L, lsh_key_dim).
            key_up (torch.Tensor): Key tensor after up-projection of shape (B, L, lsh_key_dim).
            head_idx (int): Index of the attention head.

        Returns:
            torch.Tensor: Candidate indices tensor of shape (B, L, k_max).
        """
        B, L, _ = query_up.size()
        device = query_up.device

        start_time = time.time()

        # Split features by dimension groups
        query_groups = self.split_features_by_dim_groups(query_up) # List of (B, L, group_dim) tensors
        key_groups = self.split_features_by_dim_groups(key_up) # List of (B, L, group_dim) tensors

        # Process each dimension group in parallel using torch.jit if available
        cand_list = []
        for q_grp, k_grp in zip(query_groups, key_groups):
            cand_list.append(self._process_dimension_group_candidates(q_grp, k_grp, head_idx)) # List of (B, L, k_max) tensors

        # Merge candidates from all groups with optimized unique operation
        if cand_list:
            # Use torch.cat for more efficient concatenation
            merged = torch.cat(cand_list, dim=-1) # (B, L, total_candidates)
            # Sort once for better cache locality
            merged, _ = torch.sort(merged, dim=-1)
            # Use more efficient uniqueness operation
            result = torch.unique_consecutive(merged, dim=-1)[:, :, :self.config.k_max] # (B, L, k_max)
        else:
            result = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=device) # (B, L, k_max)
        check_for_nan(result, "CandidateFinder output") # NaN check

        processing_time = time.time() - start_time
        if self.training and head_idx == 0:  # Log only for first head to reduce spam
            logger.debug(f"Candidate finding took {processing_time:.4f}s")

        return result

    def get_cache_stats(self):
        """
        Return cache statistics for monitoring.

        Returns:
            Dict: Dictionary containing cache hit/miss statistics.
        """
        return {
            "wu_manber_hits": self.wu_manber_cache_hits,
            "wu_manber_misses": self.wu_manber_cache_misses,
            "trie_cache_size": len(self.trie_cache.cache)
        }

class AbsorptionProjection(nn.Module):
    """
    Improved AbsorptionProjection with optimized matrix operations.

    Args:
        query_dim (int): Dimension of the query vectors.
        key_dim (int): Dimension of the key vectors.
        rank (int): Rank of the low-rank approximation.

    Input:
        query (torch.Tensor): Query tensor of shape (B, L, query_dim).
        key (torch.Tensor): Key tensor of shape (B, L, key_dim).

    Output:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing projected query and original key.
            - Projected query: torch.Tensor of shape (B, L, key_dim).
            - Original key: torch.Tensor of shape (B, L, key_dim).
    """
    def __init__(self, query_dim: int, key_dim: int, rank: int):
        super().__init__()
        # Better initialization for stability
        std = 1.0 / math.sqrt(rank)
        self.u_q = nn.Parameter(torch.randn(query_dim, rank) * std) # (query_dim, rank)
        self.v_q = nn.Parameter(torch.randn(rank, key_dim) * std) # (rank, key_dim)
        self.u_k = nn.Parameter(torch.randn(key_dim, rank) * std) # (key_dim, rank)
        self.v_k = nn.Parameter(torch.randn(rank, key_dim) * std) # (rank, key_dim)

        # Cache for composed matrices
        self.register_buffer('W_absorb', None)
        self.needs_composition = True

    def _compute_absorption_matrix(self):
        """Compute and cache the absorption matrix."""
        W_UQ = torch.matmul(self.u_q, self.v_q) # (query_dim, key_dim) = (query_dim, rank) @ (rank, key_dim)
        W_UK = torch.matmul(self.u_k, self.v_k) # (key_dim, key_dim) = (key_dim, rank) @ (rank, key_dim)
        self.W_absorb = torch.matmul(W_UK.transpose(0, 1), W_UQ) # (key_dim, query_dim) = (key_dim, key_dim) @ (query_dim, key_dim).T
        self.needs_composition = False
        check_for_nan(self.W_absorb, "Absorption Matrix W_absorb") # NaN check

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply absorption projection efficiently.

        Args:
            query (torch.Tensor): Query tensor of shape (B, L, query_dim).
            key (torch.Tensor): Key tensor of shape (B, L, key_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing projected query and original key.
                - Projected query: torch.Tensor of shape (B, L, key_dim).
                - Original key: torch.Tensor of shape (B, L, key_dim).
        """
        # Compute and cache absorption matrix if needed
        if self.needs_composition or self.W_absorb is None:
            self._compute_absorption_matrix()

        # Project query using cached matrix
        Q_proj = torch.matmul(query, self.W_absorb.transpose(0, 1)) # (B, L, key_dim) = (B, L, query_dim) @ (key_dim, query_dim).T
        check_for_nan(Q_proj, "Absorption Projection Q_proj") # NaN check

        return Q_proj, key

    def train(self, mode: bool = True):
        """Override train to reset composition cache when training."""
        if mode and not self.training:
            self.needs_composition = True
        return super().train(mode)

class FastAttention(nn.Module):
    """
    Improved FastAttention with optimized computations and better parallelism.

    Args:
        config (FastAttentionConfig): Configuration for FastAttention.

    Input:
        query (torch.Tensor): Query tensor of shape (B, L, d_model).
        key (torch.Tensor): Key tensor of shape (B, L, d_model).
        value (torch.Tensor): Value tensor of shape (B, L, d_model).
        mask (Optional[torch.Tensor]): Attention mask of shape (B, L, L).

    Output:
        torch.Tensor: Output tensor of shape (B, L, d_model).
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config

        # Projections for queries and key-values
        self.query_down_proj = nn.Linear(config.d_model, config.d_query) # (d_model, d_query)
        self.key_value_down_proj = nn.Linear(config.d_model, config.d_key) # (d_model, d_key)

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
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model) # (d_model * n_heads, d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Performance tracking
        self.forward_times = []

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optimized computations and memory usage.

        Args:
            query (torch.Tensor): Query tensor of shape (B, L, d_model).
            key (torch.Tensor): Key tensor of shape (B, L, d_model).
            value (torch.Tensor): Value tensor of shape (B, L, d_model).
            mask (Optional[torch.Tensor]): Attention mask of shape (B, L, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model).
        """
        start_time = time.time()

        # Validate inputs
        validate_tensor_dimensions(query, "query", query.dim())
        validate_tensor_dimensions(key, "key", key.dim())
        validate_tensor_dimensions(value, "value", value.dim())

        B, L, _ = query.size()
        device = query.device

        # Project inputs (reuse memory with inplace operations where possible)
        query_down = self.query_down_proj(query) # (B, L, d_query) = (B, L, d_model) @ (d_model, d_query)
        key_down = self.key_value_down_proj(key) # (B, L, d_key) = (B, L, d_model) @ (d_model, d_key)
        check_for_nan(query_down, "Query Down Projection") # NaN check
        check_for_nan(key_down, "Key Down Projection") # NaN check


        # Pre-allocate output tensor for concatenating head outputs
        concat = torch.zeros(B, L, self.config.d_model * self.config.n_heads, device=device) # (B, L, d_model * n_heads)

        # Process each attention head
        for head_idx in range(self.config.n_heads):
            # Apply absorption projection
            Q_proj, K_proj = self.absorption_projs[head_idx](query_down, key_down) # Q_proj: (B, L, d_key), K_proj: (B, L, d_key)
            check_for_nan(Q_proj, f"Head {head_idx} Absorption Projection Q_proj") # NaN check
            check_for_nan(K_proj, f"Head {head_idx} Absorption Projection K_proj") # NaN check


            # Find attention candidates
            candidates = self.candidate_finder(Q_proj, K_proj, head_idx) # (B, L, k_max)
            check_for_nan(candidates, f"Head {head_idx} CandidateFinder output") # NaN check


            # Create mask for valid candidates
            cand_mask = candidates != -1 # (B, L, k_max) (bool)

            # Handle invalid indices safely
            safe_candidates = candidates.clone() # (B, L, k_max)
            safe_candidates.masked_fill_(~cand_mask, 0)  # In-place operation

            # Get dimensions
            num_candidates = candidates.size(-1) # k_max

            # Create batch indices for gathering efficiently
            b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, L, num_candidates) # (B, L, k_max)

            # Gather candidate keys
            candidate_keys = K_proj[b_idx, safe_candidates] # (B, L, k_max, d_key)
            check_for_nan(candidate_keys, f"Head {head_idx} Candidate Keys") # NaN check


            # Apply Random Fourier Features if enabled
            q_exp = Q_proj.unsqueeze(2) # (B, L, 1, d_key)
            if self.config.use_rff:
                rff_encoder = self.rff_encoders[head_idx]
                q_exp = rff_encoder(q_exp) # (B, L, 1, rff_dim)
                candidate_keys = rff_encoder(candidate_keys) # (B, L, k_max, rff_dim)
                scale = optimized_sqrt(q_exp.size(-1)) # sqrt(rff_dim)
            else:
                scale = optimized_sqrt(self.config.d_key) # sqrt(d_key)
            check_for_nan(q_exp, f"Head {head_idx} Q_exp after RFF") # NaN check
            check_for_nan(candidate_keys, f"Head {head_idx} Candidate Keys after RFF") # NaN check


            # Compute attention scores with optimized batch matmul
            sim = torch.matmul(q_exp, candidate_keys.transpose(-2, -1)).squeeze(2) / scale # (B, L, k_max) = (B, L, 1, rff_dim) @ (B, L, k_max, rff_dim).transpose(-2, -1) -> (B, L, 1, k_max) -> (B, L, k_max)
            check_for_nan(sim, f"Head {head_idx} Similarity Scores (sim)") # NaN check


            # Apply masks with in-place operations
            sim.masked_fill_(~cand_mask, -1e9) # -inf -> -1e9

            # Apply optional attention mask
            if mask is not None:
                # Expand mask for broadcasting
                expanded_mask = mask.unsqueeze(1).expand_as(sim) # (B, 1, L) -> (B, L, L) -> (B, L, k_max)
                sim.masked_fill_(~expanded_mask, -1e9) # -inf -> -1e9

            # Compute attention weights
            attn_weights = F.softmax(sim, dim=-1) # (B, L, k_max)
            attn_weights = self.dropout(attn_weights) # (B, L, k_max)
            check_for_nan(sim, f"Head {head_idx} Sim before Softmax") # sim の NaN チェックを追加
            check_for_nan(attn_weights, f"Head {head_idx} Attention Weights") # NaN check


            # Gather candidate values
            candidate_values = key_down[b_idx, safe_candidates] # (B, L, k_max, d_key)
            check_for_nan(candidate_values, f"Head {head_idx} Candidate Values") # NaN check


            # Project values to output dimension and reshape in one operation
            candidate_values = self.value_up_projs[head_idx](
                candidate_values.reshape(-1, self.config.d_key) # (B*L*k_max, d_key) -> (B*L*k_max, d_model)
            ).reshape(B, L, num_candidates, self.config.d_model) # (B, L, k_max, d_model)
            check_for_nan(candidate_values, f"Head {head_idx} Candidate Values Up-Projected") # NaN check


            # Apply attention weights to values efficiently
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values, dim=2) # (B, L, d_model) = sum((B, L, k_max, 1) * (B, L, k_max, d_model), dim=2)
            check_for_nan(head_out, f"Head {head_idx} Output (head_out)") # NaN check

            # Store directly into pre-allocated tensor
            concat_slice = slice(head_idx * self.config.d_model, (head_idx + 1) * self.config.d_model)
            concat[:, :, concat_slice] = head_out # (B, L, d_model)

        check_for_nan(concat, "Concat output before output_proj") # NaN check

        # Apply output projection and dropout
        output = self.output_proj(concat) # (B, L, d_model) = (B, L, d_model * n_heads) @ (d_model * n_heads, d_model)
        output = self.dropout(output) # (B, L, d_model)
        check_for_nan(output, "FastAttention final output") # NaN check


        # Track performance
        forward_time = time.time() - start_time
        self.forward_times.append(forward_time)
        if len(self.forward_times) > 100:  # Keep only recent times
            self.forward_times = self.forward_times[-100:]

        if self.training and torch.rand(1).item() < 0.01:  # Log occasionally during training
            avg_time = sum(self.forward_times) / len(self.forward_times)
            logger.debug(f"FastAttention forward: {avg_time:.4f}s, batch={B}, seq_len={L}")

        return output

class FeedForwardNetwork(nn.Module):
    """
    Improved FeedForwardNetwork with GeLU and potential layer normalization.

    Args:
        config (FastAttentionConfig): Configuration for FastAttention.

    Input:
        x (torch.Tensor): Input tensor of shape (B, L, d_model).

    Output:
        torch.Tensor: Output tensor of shape (B, L, d_model).
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.intermediate_dim) # (d_model, intermediate_dim)
        self.linear2 = nn.Linear(config.intermediate_dim, config.d_model) # (intermediate_dim, d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Optional layer norm for more stable gradients
        self.use_layer_norm = True # Layer Normalization を有効にする
        if self.use_layer_norm:
            self.norm = nn.LayerNorm(config.intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with improved activations and optional normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model).
        """
        x = self.linear1(x) # (B, L, intermediate_dim) = (B, L, d_model) @ (d_model, intermediate_dim)
        check_for_nan(x, "FFN linear1 output") # NaN check
        x = F.gelu(x)  # Use GeLU for smoother gradients
        if self.use_layer_norm:
            x = self.norm(x) # (B, L, intermediate_dim)
            check_for_nan(x, "FFN layer norm output") # NaN check
        x = self.dropout(x)
        x = self.linear2(x) # (B, L, d_model) = (B, L, intermediate_dim) @ (intermediate_dim, d_model)
        check_for_nan(x, "FFN linear2 output") # NaN check
        x = self.dropout(x)
        return x

class FastAttentionEncoderLayer(nn.Module):
    """
    Improved FastAttentionEncoderLayer with Pre-Norm and residual connections.

    Args:
        config (FastAttentionConfig): Configuration for FastAttention.

    Input:
        src (torch.Tensor): Source tensor of shape (B, L, d_model).
        src_mask (Optional[torch.Tensor]): Source mask of shape (B, L, L).

    Output:
        torch.Tensor: Output tensor of shape (B, L, d_model).
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.self_attn = FastAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with Pre-Norm and residual connections.

        Args:
            src (torch.Tensor): Source tensor of shape (B, L, d_model).
            src_mask (Optional[torch.Tensor]): Source mask of shape (B, L, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model).
        """
        # Pre-Norm configuration
        x = self.norm1(src) # (B, L, d_model)
        check_for_nan(x, "Encoder Layer norm1 output") # NaN check
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        check_for_nan(attn_output, "Encoder Self Attention output") # NaN check
        x = src + self.dropout(attn_output)  # Residual connection, x: (B, L, d_model)
        x = self.norm2(x) # (B, L, d_model)
        check_for_nan(x, "Encoder Layer norm2 output") # NaN check
        ff_output = self.feed_forward(x)
        check_for_nan(ff_output, "Encoder FeedForward output") # NaN check
        x = x + self.dropout(ff_output)  # Residual connection, x: (B, L, d_model)
        return x

##############################################
# SPGPO Components - Refactored and Modularized
##############################################

class ResponseGenerator(nn.Module): # 修正: nn.Module を継承
    """
    Generates responses from the policy network.

    Args:
        policy_network (FastAttentionEncoderLayer): Policy network to generate responses.
        k_responses (int): Number of responses to generate per prompt.
        temperature (float): Sampling temperature for response generation.
    """
    def __init__(self, policy_network: FastAttentionEncoderLayer, k_responses: int, temperature: float = 1.0):
        super().__init__() # 修正: super().__init__() を呼び出す
        self.policy_network = policy_network
        self.k_responses = k_responses
        self.temperature = temperature  # Add temperature

    def generate_responses(self, prompt: torch.Tensor) -> torch.Tensor:
        """
        Generates multiple responses (batched).

        Args:
            prompt (torch.Tensor): Prompt tensor of shape (B, L, d_model).

        Returns:
            torch.Tensor: Response tokens tensor of shape (B, k_responses, L).
        """
        B, L, _ = prompt.size()
        prompt_expanded = prompt.repeat_interleave(self.k_responses, dim=0) # (B*k_responses, L, d_model)
        check_for_nan(prompt_expanded, "ResponseGenerator prompt_expanded") # NaN check
        logits = self.policy_network(prompt_expanded) / self.temperature # (B*k_responses, L, vocab_size) / temperature
        check_for_nan(logits, "ResponseGenerator logits") # NaN check


        # Use Categorical distribution for sampling.
        probs = F.softmax(logits, dim=-1) # (B*k_responses, L, vocab_size)
        check_for_nan(probs, "ResponseGenerator probs before Categorical") # NaN check

        dist = Categorical(probs)
        response_tokens = dist.sample() # (B*k_responses, L)

        return response_tokens.view(B, self.k_responses, -1) # (B, k_responses, L)


class PreferenceComputer:
    """
    Computes group preferences.

    Args:
        reward_model (Callable): Reward model function.
    """
    def __init__(self, reward_model: Callable):
        self.reward_model = reward_model

    def compute_group_preference(self, response: torch.Tensor, response_group: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Computes batched group preference P(y ≻ G_x\{y}|x).

        Args:
            response (torch.Tensor): Response tensor of shape (B, L).
            response_group (torch.Tensor): Response group tensor of shape (B, K, L).
            prompt (torch.Tensor): Prompt tensor of shape (B, prompt_dim).

        Returns:
            torch.Tensor: Group preference probabilities of shape (B, K).
        """
        B, K, L = response_group.shape
        response_expanded = response.unsqueeze(1)  # (B, 1, L)
        response_group_expanded = response_group.unsqueeze(0) # (1, K, L)
        prompt_expanded = prompt.unsqueeze(1) # (B, 1, prompt_dim)
        preferences = preference_oracle(response_expanded, response_group_expanded, prompt_expanded, self.reward_model) # (B, K)
        return preferences

class AdvantageComputer:
    """
    Computes SPGPO advantage.

    Args:
        preference_computer (PreferenceComputer): Preference computer instance.
        k_responses (int): Number of responses per prompt.
    """
    def __init__(self, preference_computer: PreferenceComputer, k_responses: int):
        self.preference_computer = preference_computer
        self.k_responses = k_responses

    def compute_spgpo_advantage(self, response: torch.Tensor, response_group: torch.Tensor, prompt: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        """
        Computes batched SPGPO advantage A_SPGPO(y|x) with baseline using vectorized operations.

        Args:
            response (torch.Tensor): Response tensor of shape (B, L).
            response_group (torch.Tensor): Response group tensor of shape (B, K, L).
            prompt (torch.Tensor): Prompt tensor of shape (B, prompt_dim).
            baseline (torch.Tensor): Baseline tensor of shape (B,).

        Returns:
            torch.Tensor: SPGPO advantage tensor of shape (B,).
        """
        B, K, L = response_group.shape
        preferences = self.preference_computer.compute_group_preference(response, response_group, prompt) # (B, K)

        # Create mask using vectorized operations instead of loop
        mask = torch.ones((B, K), device=response.device) # (B, K)
        response_expanded = response.unsqueeze(1)  # (B, 1, L)
        equals = (response_expanded == response_group).all(dim=2)  # Compare along sequence length, (B, K)
        mask.masked_fill_(equals, 0) # Mask out self-preference

        # Calculate response preference with numerical stability
        denom = torch.clamp(self.k_responses - equals.sum(dim=1, keepdim=True), min=1.0) # (B, 1)
        response_pref = (preferences * mask).sum(dim=1) / denom # (B,)

        # Calculate advantage with average group preference
        avg_group_pref = preferences.mean(dim=1) # (B,)
        advantage = response_pref - avg_group_pref - baseline # (B,)

        return advantage

class SPGPOEnvironment:
    """
    Environment for SPGPO training, orchestrating response generation, preference, and advantage computation.

    Args:
        policy_network (FastAttentionEncoderLayer): Policy network for response generation.
        prompt_distribution (Callable): Function to sample prompts.
        reward_model (Callable): Reward model function.
        k_responses (int): Number of responses to generate per prompt.
        tokenizer (Callable, optional): Tokenizer function. Defaults to None.
        clip_ratio (float): Clipping ratio for PPO-style loss.
    """
    def __init__(self, policy_network: FastAttentionEncoderLayer, prompt_distribution: Callable,
                 reward_model: Callable, k_responses: int = 4, tokenizer: Callable = None,
                 clip_ratio: float = 0.2):
        """Initializes the SPGPO environment with modular components."""
        self.prompt_distribution = prompt_distribution
        self.tokenizer = tokenizer
        self.clip_ratio = clip_ratio

        # Modular components
        self.response_generator = ResponseGenerator(policy_network, k_responses)
        self.preference_computer = PreferenceComputer(reward_model)
        self.advantage_computer = AdvantageComputer(self.preference_computer, k_responses)
        self.policy_network = policy_network # Directly use policy network for log prob calculation


        if self.tokenizer is None:
            self.tokenizer = lambda x: torch.randint(0, 100, (1, x))


    def step(self, prompts: torch.Tensor, old_log_probs_batch: torch.Tensor, baselines: torch.Tensor) -> torch.Tensor:
        """
        Performs a batched step in the SPGPO environment using modular components.

        Args:
            prompts (torch.Tensor): Batch of prompt tensors of shape (B, L, d_model).
            old_log_probs_batch (torch.Tensor): Batch of old log probabilities of shape (B*k_responses,).
            baselines (torch.Tensor): Batch of baseline tensors of shape (B,).

        Returns:
            torch.Tensor: SPGPO loss tensor (scalar).
        """
        B, _, _ = prompts.shape
        all_responses = []
        all_advantages = []
        all_current_log_probs = []

        for i in range(B):
            prompt = prompts[i].unsqueeze(0) # (1, L, d_model)
            response_group = self.response_generator.generate_responses(prompt) # (1, k_responses, L) -> (k_responses, L)
            response_group = response_group.squeeze(0) # (k_responses, L)

            logits = self.policy_network(response_group) # (k_responses, L, vocab_size)
            check_for_nan(logits, "Env Policy Network logits") # NaN check
            current_log_probs = F.log_softmax(logits, dim=-1) # (k_responses, L, vocab_size)
            check_for_nan(current_log_probs, "Env Log Probs") # NaN check

            for k in range(self.response_generator.k_responses): # Use k_responses from generator
                response = response_group[k] # (L,)
                all_responses.append(response)

                advantage = self.advantage_computer.compute_spgpo_advantage(
                    response.unsqueeze(0), response_group.unsqueeze(0), prompt, baselines[i].unsqueeze(0) # response:(1, L), response_group:(1, k_responses, L), prompt:(1, L, d_model), baseline:(1,)
                ) # (1,)
                all_advantages.append(advantage.squeeze(0)) # scalar

                response_tokens = response.long() # (L,)
                current_log_prob_response = current_log_probs[k] # (L, vocab_size)
                gathered_log_probs = torch.gather(current_log_prob_response, 1, response_tokens.unsqueeze(1)).squeeze(1) # (L,)
                summed_log_prob = gathered_log_probs.sum() # scalar
                all_current_log_probs.append(summed_log_prob) # scalar

        all_responses = torch.stack(all_responses) # (B*k_responses, L)
        all_advantages = torch.stack(all_advantages) # (B*k_responses,)
        all_current_log_probs = torch.stack(all_current_log_probs) # (B*k_responses,)

        # Add clipping to prevent extreme values
        log_diff = all_current_log_probs - old_log_probs_batch # (B*k_responses,)
        log_diff = torch.clamp(log_diff, -20, 20)  # Prevent extreme values
        ratios = torch.exp(log_diff) # (B*k_responses,)

        surr1 = ratios * all_advantages # (B*k_responses,)
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * all_advantages # (B*k_responses,)
        loss = -torch.min(surr1, surr2).mean() # scalar
        return loss


class SPGPOAgent(nn.Module):
    """
    Agent class encapsulating the policy network.

    Args:
        config (FastAttentionConfig): Configuration for FastAttention.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.policy_network = FastAttentionEncoderLayer(config)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            prompt (torch.Tensor): Prompt tensor of shape (B, L, d_model).

        Returns:
            torch.Tensor: Logits tensor of shape (B, L, vocab_size).
        """
        return self.policy_network(prompt)

def compute_baselines(prompts: torch.Tensor, env: SPGPOEnvironment) -> torch.Tensor:
    """
    Computes baselines for each prompt using average group preference.

    Args:
        prompts (torch.Tensor): Batch of prompt tensors of shape (B, L, d_model).
        env (SPGPOEnvironment): SPGPO environment instance.

    Returns:
        torch.Tensor: Baseline tensor of shape (B,).
    """
    B, _, _ = prompts.shape
    baselines = []
    for i in range(B):
        prompt = prompts[i].unsqueeze(0) # (1, L, d_model)
        response_group = env.response_generator.generate_responses(prompt) # (1, k_responses, L) -> (k_responses, L)
        response_group = response_group.squeeze(0) # (k_responses, L)
        preferences_batch = env.preference_computer.compute_group_preference(response_group, response_group.unsqueeze(0), prompt) # response_group:(k_responses, L), response_group.unsqueeze(0):(1, k_responses, L), prompt:(1, L, d_model) -> (k_responses, k_responses)

        avg_group_pref = preferences_batch.mean() # scalar
        baselines.append(avg_group_pref)
    return torch.stack(baselines) # (B,)


def spgpo_training_episode(agent: SPGPOAgent, env: SPGPOEnvironment, optimizer: optim.Optimizer,
                            scheduler: optim.lr_scheduler.LambdaLR, writer: SummaryWriter, episode: int,
                            batch_size: int = 32) -> float:
    """
    Runs a single SPGPO training episode with batched prompts and PPO-style loss.

    Args:
        agent (SPGPOAgent): SPGPO agent instance.
        env (SPGPOEnvironment): SPGPO environment instance.
        optimizer (optim.Optimizer): Optimizer instance.
        scheduler (optim.lr_scheduler.LambdaLR): Learning rate scheduler instance.
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        episode (int): Current training episode number.
        batch_size (int): Batch size.

    Returns:
        float: Total loss for the episode.
    """
    total_loss = 0.0
    prompts = env.prompt_distribution(batch_size) # (B, L, d_model)

    baselines = compute_baselines(prompts, env).detach() # (B,)

    old_log_probs_batch = []
    with torch.no_grad():
        for i in range(batch_size):
            prompt = prompts[i].unsqueeze(0) # (1, L, d_model)
            response_group = env.response_generator.generate_responses(prompt) # (1, k_responses, L) -> (k_responses, L)
            response_group = response_group.squeeze(0) # (k_responses, L)

            logits = agent(response_group) # (k_responses, L, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1) # (k_responses, L, vocab_size)

            for k in range(env.response_generator.k_responses): # Use k_responses from generator
                response = response_group[k] # (L,)
                response_tokens = response.long() # (L,)
                log_prob_response = log_probs[k] # (L, vocab_size)
                gathered_log_probs = torch.gather(log_prob_response, 1, response_tokens.unsqueeze(1)).squeeze(1) # (L,)
                summed_log_prob = gathered_log_probs.sum() # scalar
                old_log_probs_batch.append(summed_log_prob) # scalar

    old_log_probs_batch = torch.stack(old_log_probs_batch).detach() # (B*k_responses,)


    loss = env.step(prompts, old_log_probs_batch, baselines) # scalar

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0) # Gradient Clipping
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()

    writer.add_scalar("Loss/Episode", loss.item(), episode)
    return total_loss


def train_spgpo(config: FastAttentionConfig, prompt_distribution: Callable, reward_model: Callable,
                 tokenizer: Callable, num_episodes: int = 100, batch_size: int = 32, k_responses: int = 4):
    """
    Main training loop for SPGPO with batched training and TensorBoard logging.

    Args:
        config (FastAttentionConfig): Configuration for FastAttention.
        prompt_distribution (Callable): Function to sample prompts.
        reward_model (Callable): Reward model function.
        tokenizer (Callable): Tokenizer function.
        num_episodes (int): Number of training episodes.
        batch_size (int): Batch size.
        k_responses (int): Number of responses to generate per prompt.

    Returns:
        SPGPOAgent: Trained SPGPO agent instance.
    """
    agent = SPGPOAgent(config)
    optimizer = optim.AdamW(agent.parameters(), lr=3e-4, weight_decay=0.01) # Consider reducing lr if still NaN
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    writer = SummaryWriter()

    env = SPGPOEnvironment(agent.policy_network, prompt_distribution, reward_model, k_responses, tokenizer, clip_ratio=0.2) # Pass policy_network

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
        return torch.randn(batch_size, seq_len, prompt_dim) # (B, L, d_model)

    def simple_reward_model(prompt, response):
        # Dummy reward model for example
        return torch.randn(prompt.size(0)) # (B,)

    def simple_tokenizer(seq_len):
        # Dummy tokenizer
        return torch.randint(0, 100, (1, seq_len)) # (1, L)

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
