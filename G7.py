import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

##############################################
# Low-level Helpers
##############################################

def optimized_sqrt(n: int) -> float:
    """
    Optimized square root calculation for powers of 2.
    
    For powers of 2, uses bit manipulation for faster computation.
    For other values, falls back to standard math.sqrt.
    
    Args:
        n: Integer to compute square root of
    
    Returns:
        Square root of n as float
    """
    if n & (n - 1) == 0:  # Check if n is a power of 2
        k = n.bit_length() - 1
        return 2 ** (k / 2)
    return math.sqrt(n)

def fma(a: float, b: float, c: float) -> float:
    """
    Fused multiply-add operation: a*b + c.
    
    Uses hardware FMA if available, otherwise falls back to standard operations.
    
    Args:
        a: First multiplicand
        b: Second multiplicand
        c: Addend
    
    Returns:
        Result of a*b + c
    """
    try:
        return math.fma(a, b, c)  # Use hardware FMA if available
    except AttributeError:
        return a * b + c  # Fallback implementation

##############################################
# Fast Attention Configuration
##############################################

@dataclass
class FastAttentionConfig:
    """Configuration for Fast Attention components."""

    # Model dimensions
    d_model: int          # Overall model dimension
    d_key: int            # Key dimension
    d_query: int          # Query dimension
    n_heads: int          # Number of attention heads

    # Approximation parameters
    rank: int             # Rank for low-rank approximations
    rff_dim: int          # Output dimension for random Fourier features

    # Candidate search parameters
    k_max: int            # Maximum candidate keys per query
    stride: int           # Stride length for Trie
    lsh_buckets: int      # Number of LSH buckets
    lsh_bandwidth: float  # Bandwidth parameter for LSH
    lsh_key_dim: int      # LSH input dimension
    wu_manber_prefix_len: int  # Prefix length for Wu-Manber search

    # Optional parameters
    hyper_cuts_dim_groups: Optional[List[int]] = None  # Dimension groups for HyperCuts-inspired partitioning
    n_lsh_hashes: int = 4  # Number of hash functions for LSH
    dropout: float = 0.1  # Dropout probability
    intermediate_dim: int = 2048  # Dimension of intermediate FFN layer
    use_rff: bool = True  # Whether to use Random Fourier Features

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.hyper_cuts_dim_groups is not None:
            total_dims = sum(self.hyper_cuts_dim_groups)
            if total_dims != self.d_key:
                raise ValueError(f"Sum of hyper_cuts_dim_groups ({total_dims}) must equal d_key ({self.d_key})")

        if self.d_query <= 0 or self.d_key <= 0 or self.d_model <= 0:
            raise ValueError("Dimensions must be positive")

        if self.k_max <= 0:
            raise ValueError("k_max must be positive")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("Dropout must be in [0, 1)")

##############################################
# Low-Rank Projection Components
##############################################

class LowRankLinear(nn.Module):
    """
    Low-rank approximation of a linear transformation.
    
    Represents a matrix W of shape (in_features, out_features) as a product
    of two smaller matrices: W = U * V where U is (in_features, rank) and
    V is (rank, out_features).
    
    This reduces parameter count from O(in_features * out_features) to
    O((in_features + out_features) * rank).
    """

    def __init__(self, in_features: int, out_features: int, rank: int):
        """
        Initialize a low-rank linear transformation.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of the approximation
        """
        super().__init__()

        # Initialize with scaled random values for better gradient flow
        self.u_weight = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.v_weight = nn.Parameter(torch.randn(rank, out_features) / math.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the low-rank linear transformation.
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            Output tensor of shape (..., out_features)
        """
        return torch.matmul(torch.matmul(x, self.u_weight), self.v_weight)

##############################################
# Random Fourier Features
##############################################

class RandomFourierFeatures(nn.Module):
    """
    Random Fourier Features for kernel approximation.
    
    Approximates a Gaussian kernel by projecting inputs into a randomized
    feature space where dot products approximate kernel evaluations.
    """

    def __init__(self, input_dim: int, rff_dim: int):
        """
        Initialize the Random Fourier Features module.
        
        Args:
            input_dim: Dimension of input features
            rff_dim: Dimension of output features
        """
        super().__init__()

        # Random projection matrix (fixed during training)
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim), requires_grad=False)

        # Random bias terms (fixed during training)
        self.bias = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi, requires_grad=False)

        # Scale factor to normalize variance
        self.scale = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input using Random Fourier Features.
        
        Args:
            x: Input tensor of shape (..., input_dim)
        
        Returns:
            Transformed tensor of shape (..., rff_dim)
        """
        # Project and add bias
        projection = x.matmul(self.omega) + self.bias

        # Apply cosine and scale
        return torch.cos(projection) * self.scale

##############################################
# Locality-Sensitive Hashing
##############################################

class LSHTable(nn.Module):
    """
    Locality-Sensitive Hashing table for approximate nearest neighbor search.
    
    Projects vectors into discrete hash buckets such that similar vectors are
    likely to hash to the same bucket.
    """

    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int):
        """
        Initialize the LSH table.
        
        Args:
            dim: Dimension of input vectors
            n_buckets: Number of hash buckets
            bandwidth: Bandwidth parameter (controls sensitivity)
            n_hashes: Number of hash functions to use
        """
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes

        # Random projection vectors for hashing
        self.random_vectors = nn.Parameter(
            torch.randn(dim, n_hashes),
            requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hash input vectors.
        
        Args:
            x: Input tensor of shape (..., dim)
        
        Returns:
            Hash buckets of shape (..., n_hashes), each value in range [0, n_buckets-1]
        """
        # Project input onto random vectors
        proj = x.matmul(self.random_vectors)

        # Quantize and take modulo to get bucket indices
        return torch.floor(proj / self.bandwidth) % self.n_buckets

##############################################
# Trie-based Prefix Search
##############################################

_TRIE_INDICES_KEY = '_indices'

class Trie(nn.Module):
    """
    Trie data structure for efficient prefix matching.
    
    Stores binary vector prefixes for fast candidate retrieval.
    """

    def __init__(self, stride: int):
        """
        Initialize the Trie.
        
        Args:
            stride: Stride length for prefix chunks
        """
        super().__init__()
        self.root_node: Dict[Any, Any] = {}
        self.stride_len = stride

    def insert(self, binary_vector: torch.Tensor, index: int) -> None:
        """
        Insert a binary vector into the Trie.
        
        Args:
            binary_vector: Binary vector to insert
            index: Index associated with this vector
        """
        current_node = self.root_node

        # Process vector in chunks of size stride_len
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())

            # Create node if it doesn't exist
            if prefix not in current_node:
                current_node[prefix] = {}

            current_node = current_node[prefix]

        # Store index at leaf node
        current_node.setdefault(_TRIE_INDICES_KEY, []).append(index)

    def search(self, binary_vector: torch.Tensor) -> List[int]:
        """
        Search for matching prefixes in the Trie.
        
        Args:
            binary_vector: Binary vector to search for
        
        Returns:
            List of indices for matching prefixes
        """
        current_node = self.root_node

        # Follow path through Trie
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())

            if prefix not in current_node:
                return []  # No match found

            current_node = current_node[prefix]

        # Return stored indices
        return current_node.get(_TRIE_INDICES_KEY, [])

##############################################
# Candidate Key Search
##############################################

class CandidateFinder(nn.Module):
    """
    Finds candidate keys for each query using multiple search strategies.
    
    Combines Wu-Manber algorithm, Trie prefix matching, and
    HyperCuts-inspired dimension partitioning.
    """

    def __init__(self, config: FastAttentionConfig, tries: List[Trie], lsh_tables: nn.ModuleList):
        """
        Initialize the CandidateFinder.
        
        Args:
            config: Fast attention configuration
            tries: List of Tries for each attention head
            lsh_tables: List of LSH tables for each attention head
        """
        super().__init__()
        self.config = config
        self.tries = tries
        self.lsh_tables = lsh_tables
        self.wu_manber_prefix_len = config.wu_manber_prefix_len
        self.hyper_cuts_dim_groups = config.hyper_cuts_dim_groups

    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert features to binary representation.
        
        Args:
            x: Input tensor
        
        Returns:
            Binary tensor (0.0 or 1.0 values)
        """
        return (x > 0).float()

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Split features along dimension groups for HyperCuts-inspired search.
        
        Args:
            features: Input features of shape (batch, length, dim)
        
        Returns:
            List of tensors, each containing a subset of dimensions
        """
        if self.hyper_cuts_dim_groups is None:
            return [features]

        groups = []
        start = 0

        for group_dim in self.hyper_cuts_dim_groups:
            groups.append(features[:, :, start:start+group_dim])
            start += group_dim

        return groups

    def _build_wu_manber_hash_table(self, key_bin: torch.Tensor) -> Dict[tuple, List[int]]:
        """
        Build Wu-Manber hash table from binary keys.
        
        Args:
            key_bin: Binary key tensor of shape (length, dim)
        
        Returns:
            Dictionary mapping prefixes to lists of indices
        """
        table: Dict[tuple, List[int]] = {}
        L = key_bin.size(0)

        for i in range(L):
            prefix = tuple(key_bin[i, :self.config.wu_manber_prefix_len].tolist())
            table.setdefault(prefix, []).append(i)

        return table

    def _wu_manber_search(self, query_bin: torch.Tensor, table: Dict[tuple, List[int]]) -> List[int]:
        """
        Search Wu-Manber hash table for matching prefixes.
        
        Args:
            query_bin: Binary query tensor
            table: Wu-Manber hash table
        
        Returns:
            List of matching indices
        """
        prefix = tuple(query_bin[:self.config.wu_manber_prefix_len].tolist())
        return table.get(prefix, [])

    def _get_wu_manber_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor) -> List[List[List[int]]]:
        """
        Get Wu-Manber candidates for a dimension group.
        
        Args:
            query_grp: Query features for this dimension group
            key_grp: Key features for this dimension group
        
        Returns:
            List of candidate lists for each batch and position
        """
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
        """
        Get Trie-based candidates for a dimension group.
        
        Args:
            query_grp: Query features for this dimension group
            key_grp: Key features for this dimension group
            head_idx: Attention head index
        
        Returns:
            List of candidate lists for each batch and position
        """
        B, L, _ = key_grp.size()
        cand_lists = []

        for b in range(B):
            trie = Trie(self.config.stride)
            key_bin = self.binary_quantize(key_grp[b])

            for i in range(L):
                trie.insert(key_bin[i], i)

            batch_list = [trie.search(self.binary_quantize(query_grp[b][i])) for i in range(L)]
            cand_lists.append(batch_list)

        return cand_lists

    def merge_candidate_indices_groups(self, cand_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge candidate indices from multiple dimension groups.
        
        Args:
            cand_tensors: List of candidate tensors
        
        Returns:
            Merged and deduplicated tensor of candidates
        """
        merged = torch.cat(cand_tensors, dim=-1)
        merged, _ = torch.sort(merged)
        return torch.unique_consecutive(merged, dim=-1)

    def _process_dimension_group_candidates(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        Process candidates for a dimension group.
        
        Args:
            query_grp: Query features for this dimension group
            key_grp: Key features for this dimension group
            head_idx: Attention head index
        
        Returns:
            Tensor of candidate indices
        """
        B, L, _ = query_grp.size()

        # Get candidates from both search methods
        wu_cands = self._get_wu_manber_candidates_group(query_grp, key_grp)
        trie_cands = self._get_trie_candidates_group(query_grp, key_grp, head_idx)

        # Initialize output tensor
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_grp.device)

        # Take intersection of candidates and limit to k_max
        for b in range(B):
            for i in range(L):
                common = list(set(wu_cands[b][i]) & set(trie_cands[b][i]))

                if common:
                    common_tensor = torch.tensor(common, dtype=torch.long, device=query_grp.device)

                    if common_tensor.numel() > self.config.k_max:
                        common_tensor = common_tensor[:self.config.k_max]

                    candidates[b, i, :common_tensor.numel()] = common_tensor

        return candidates

    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        Find candidate keys for each query.
        
        Args:
            query_up: Query features
            key_up: Key features
            head_idx: Attention head index
        
        Returns:
            Tensor of candidate indices
        """
        B, L, _ = query_up.size()

        # Split by dimension groups
        query_groups = self.split_features_by_dim_groups(query_up)
        key_groups = self.split_features_by_dim_groups(key_up)

        # Process each dimension group
        cand_list = [
            self._process_dimension_group_candidates(q_grp, k_grp, head_idx)
            for q_grp, k_grp in zip(query_groups, key_groups)
        ]

        # Merge candidates from all groups
        if cand_list:
            merged = self.merge_candidate_indices_groups(cand_list)
            return merged[:, :, :self.config.k_max]

        # Fallback if no candidates found
        return torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_up.device)

##############################################
# Absorption Projection for Attention
##############################################

class AbsorptionProjection(nn.Module):
    """
    Absorption Projection for efficient attention computation.
    
    Projects queries and keys into a compatible space using low-rank
    approximations of weight matrices.
    """

    def __init__(self, query_dim: int, key_dim: int, rank: int):
        """
        Initialize the Absorption Projection.
        
        Args:
            query_dim: Query dimension
            key_dim: Key dimension
            rank: Rank for low-rank approximations
        """
        super().__init__()

        # Low-rank factors for query projection
        self.u_q = nn.Parameter(torch.randn(query_dim, rank) / math.sqrt(rank))
        self.v_q = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))

        # Low-rank factors for key projection
        self.u_k = nn.Parameter(torch.randn(key_dim, rank) / math.sqrt(rank))
        self.v_k = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply absorption projection to query and key.
        
        Args:
            query: Query tensor
            key: Key tensor
        
        Returns:
            Tuple of (projected_query, key)
        """
        # Reconstruct weight matrices from low-rank factors
        W_UQ = torch.matmul(self.u_q, self.v_q)
        W_UK = torch.matmul(self.u_k, self.v_k)

        # Compute absorption matrix
        W_absorb = torch.matmul(W_UK.transpose(0, 1), W_UQ)

        # Project query through absorption matrix
        Q_proj = torch.matmul(query, W_absorb.transpose(0, 1))

        return Q_proj, key

##############################################
# Fast Attention Implementation
##############################################

class FastAttention(nn.Module):
    """
    Fast Attention implementation combining multiple efficiency techniques.
    
    Uses candidate search, low-rank projections, and optional random Fourier features
    to approximate full attention with reduced computational complexity.
    """

    def __init__(self, config: FastAttentionConfig):
        """
        Initialize the Fast Attention module.
        
        Args:
            config: Fast attention configuration
        """
        super().__init__()
        self.config = config

        # Input projections
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

        # Random Fourier Feature encoders for each head
        self.rff_encoders = nn.ModuleList([
            RandomFourierFeatures(config.d_key, config.rff_dim)
            for _ in range(config.n_heads)
        ])

        # LSH tables for each head and dimension group
        if config.hyper_cuts_dim_groups:
            self.lsh_tables_list = nn.ModuleList([
                nn.ModuleList([
                    LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
                    for dim in config.hyper_cuts_dim_groups
                ])
                for _ in range(config.n_heads)
            ])
        else:
            self.lsh_tables_list = nn.ModuleList([
                nn.ModuleList([
                    LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
                ])
                for _ in range(config.n_heads)
            ])

        # Tries for each head
        self.tries_list = nn.ModuleList([
            Trie(config.stride) for _ in range(config.n_heads)
        ])

        # Candidate finder
        self.candidate_finder = CandidateFinder(
            config,
            list(self.tries_list),
            self.lsh_tables_list
        )

        # Output projection
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply fast attention.
        
        Args:
            query: Query tensor of shape (batch, length, d_model)
            key: Key tensor of shape (batch, length, d_model)
            value: Value tensor of shape (batch, length, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch, length, d_model)
        """
        B, L, _ = query.size()

        # Project inputs to lower dimensions
        query_down = self.query_down_proj(query)
        key_down = self.key_value_down_proj(key)

        head_outputs = []

        # Process each attention head
        for head_idx in range(self.config.n_heads):
            # Apply absorption projection
            Q_proj, K_proj = self.absorption_projs[head_idx](query_down, key_down)

            # Find candidate keys for each query
            candidates = self.candidate_finder(Q_proj, K_proj, head_idx)
            cand_mask = candidates != -1

            # Replace invalid indices with 0 to avoid indexing errors
            safe_candidates = candidates.clone()
            safe_candidates[safe_candidates == -1] = 0

            num_candidates = candidates.size(-1)

            # Create batch indices for gathering
            b_idx = torch.arange(B, device=K_proj.device).view(B, 1, 1).expand(B, L, num_candidates)

            # Gather candidate keys
            candidate_keys = K_proj[b_idx, safe_candidates]

            # Prepare query for similarity computation
            q_exp = Q_proj.unsqueeze(2)

            # Apply Random Fourier Features if enabled
            if self.config.use_rff:
                q_exp = self.rff_encoders[head_idx](q_exp.reshape(-1, self.config.d_key)).reshape(B, L, 1, self.config.rff_dim)
                candidate_keys = self.rff_encoders[head_idx](candidate_keys.reshape(-1, self.config.d_key)).reshape(B, L, num_candidates, self.config.rff_dim)
                scale = optimized_sqrt(self.config.rff_dim)
            else:
                scale = optimized_sqrt(self.config.d_key)

            # Compute similarity scores
            sim = torch.matmul(q_exp, candidate_keys.transpose(-2, -1)).squeeze(2) / scale

            # Mask invalid candidates
            sim = sim.masked_fill(~cand_mask, float('-inf'))

            # Compute attention weights
            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Gather candidate values
            candidate_values = key_down[b_idx, safe_candidates]

            # Project values to output dimension
            candidate_values = self.value_up_projs[head_idx](candidate_values.reshape(-1, self.config.d_key)).reshape(B, L, num_candidates, self.config.d_model)

            # Apply attention weights to values
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values, dim=2)

            head_outputs.append(head_out)

        # Concatenate outputs from all heads
        concat = torch.cat(head_outputs, dim=-1)

        # Final output projection
        output = self.output_proj(concat)

        return self.dropout(output)

##############################################
# Feed-Forward Network
##############################################

class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network for Transformer-like architectures.
    
    Standard two-layer FFN with ReLU activation.
    """

    def __init__(self, config: FastAttentionConfig):
        """
        Initialize the Feed-Forward Network.
        
        Args:
            config: Fast attention configuration
        """
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.intermediate_dim)
        self.linear2 = nn.Linear(config.intermediate_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Feed-Forward Network.
        
        Args:
            x: Input tensor
        
        Returns:
            Transformed tensor
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

##############################################
# Encoder Layer
##############################################

class FastAttentionEncoderLayer(nn.Module):
    """
    Encoder layer using Fast Attention.
    
    Follows the standard Transformer encoder architecture with
    pre-layer normalization and residual connections.
    """

    def __init__(self, config: FastAttentionConfig):
        """
        Initialize the Fast Attention Encoder Layer.
        
        Args:
            config: Fast attention configuration
        """
        super().__init__()
        self.self_attn = FastAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply the encoder layer.
        
        Args:
            src: Input tensor
            src_mask: Optional attention mask
        
        Returns:
            Transformed tensor
        """
        # First sub-layer: Self-attention with residual connection
        residual = src
        src = self.norm1(src)
        attn = self.self_attn(src, src, src, mask=src_mask)
        src = residual + self.dropout(attn)

        # Second sub-layer: Feed-forward network with residual connection
        residual = src
        src = self.norm2(src)
        ffn = self.feed_forward(src)
        return residual + self.dropout(ffn)

##############################################
# Policy Network with Fast Attention Integration
##############################################

class PolicyNetworkWithAttention(nn.Module):
    """
    FastAttention を利用した状態処理を統合するポリシーネットワーク。
    
    入力状態が [batch, seq_len, state_dim] で与えられる場合、
    FastAttention を利用して抽出された特徴 [batch, seq_len, d_model] を
    集約して [batch, d_model] にし、MLP により行動分布の平均と対数標準偏差を出力する。
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 attention_config: Optional[FastAttentionConfig] = None,
                 seq_len: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.attention = FastAttentionEncoderLayer(attention_config) if attention_config is not None else None
        self.effective_state_dim = attention_config.d_model if self.attention is not None else state_dim

        self.net = nn.Sequential(
            nn.Linear(self.effective_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)  # 出力: [mean, log_std]
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attention is not None:
            # 状態を [batch, seq_len, state_dim] として処理
            batch_size = state.size(0) // self.seq_len
            state = state.view(batch_size, self.seq_len, -1)
            state = self.attention(state)
            state = state.mean(dim=1)  # [batch, d_model]
        else:
            state = state.squeeze(1)  # [batch, state_dim]

        out = self.net(state)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        std = log_std.exp()
        return mean, std

##############################################
# SPPO Agent with Fast Attention Integration
##############################################

class SPPO:
    """
    SPPO エージェント: FastAttention を利用した状態処理、
    グループ相対報酬の計算、Adam による更新、
    オンラインハイパーパラメータ調整を統合する。
    
    主な処理フロー:
      1. 状態の正規化と FastAttention による特徴抽出
      2. ポリシーネットワークからアクションのサンプリング (再パラメータ化)
      3. 報酬モデルを用いて各候補の報酬を計算し、グループ内で正規化
      4. PPO 損失の計算：クリップされた代理目的関数、価値関数損失、エントロピーボーナス
      5. Adam によるパラメータ更新
      6. エピソード終了後にベースライン更新とハイパーパラメータのオンライン調整
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy_lr: float = 1e-3,
        eta: float = 0.01,
        alpha: float = 0.1,
        num_samples: int = 8,
        correction: float = None,
        use_adam_optimizer: bool = True,
        attention_config: Optional[FastAttentionConfig] = None,
        seq_len: int = 1
    ):
        # ポリシーネットワークの初期化 (FastAttention 統合版)
        self.policy = PolicyNetworkWithAttention(state_dim, action_dim, attention_config=attention_config, seq_len=seq_len)
        # ベースラインポリシーは定期的に更新するため、初期状態は同一にする
        self.baseline_policy = PolicyNetworkWithAttention(state_dim, action_dim, attention_config=attention_config, seq_len=seq_len)
        self.baseline_policy.load_state_dict(self.policy.state_dict())

        self.eta = eta
        self.alpha = alpha
        self.num_samples = num_samples
        self.correction = correction or eta / 2
        self.use_adam_optimizer = use_adam_optimizer

        if use_adam_optimizer:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        else:
            self.optimizer = None  # Mirror Descent 更新を用いる

    def normalize_state(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        状態テンソルを各サンプルごとにゼロ平均・単位分散に正規化する。
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + eps)

    def sample_actions(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ポリシーネットワークからアクションをサンプリングする。
        再パラメータ化トリックを使用して、ランダムなサンプルを生成する。

        Returns:
            actions: サンプリングされたアクション [batch, action_dim]
            mean: 平均 [batch, action_dim]
            std:  標準偏差 [batch, action_dim]
        """
        mean, std = self.policy(state)
        epsilon = torch.randn_like(mean)
        actions = mean + std * epsilon
        return actions, mean, std

    def compute_log_prob(self, actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        正規分布におけるアクションの対数確率を計算する。

        Returns:
            対数確率 [batch]
        """
        return (-0.5 * ((actions - mean) / std).pow(2) - std.log() - 0.5 * np.log(2 * np.pi)).sum(-1)

    def compute_group_rewards(self, state: torch.Tensor, actions: torch.Tensor, reward_model: Callable) -> torch.Tensor:
        """
        グループ内報酬の計算:
          - 各サンプルごとに報酬を計算し、ミニバッチ内で正規化する。
        
        Args:
            state: [group_size, state_dim]
            actions: [group_size, action_dim]
            reward_model: 報酬計算関数 (例: realistic_reward_model)
        
        Returns:
            グループ内相対報酬 [group_size, 1]
        """
        raw_rewards = reward_model(state, actions)  # [group_size]
        mean_reward = raw_rewards.mean()
        std_reward = raw_rewards.std() + 1e-8
        normalized_rewards = (raw_rewards - mean_reward) / std_reward
        return normalized_rewards.unsqueeze(1)

    def update_baseline(self):
        """
        エピソード終了時に、現在のポリシーネットワークのパラメータを
        ベースラインポリシーとして更新する。
        """
        self.baseline_policy.load_state_dict(self.policy.state_dict())

    def compute_ppo_loss(self, state: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """
        PPO 損失の計算:
        loss = -min(ratio * advantage, clip(ratio, 1 - eps, 1 + eps) * advantage) + value_loss + entropy_bonus
        """
        # Compute current policy log prob
        current_mean, current_std = self.policy(state)
        current_log_prob = self.compute_log_prob(actions, current_mean, current_std)

        # Compute baseline policy log prob
        with torch.no_grad():
            baseline_mean, baseline_std = self.baseline_policy(state)
            baseline_log_prob = self.compute_log_prob(actions, baseline_mean, baseline_std)

        # Compute ratio
        ratio = torch.exp(current_log_prob - baseline_log_prob)

        # Compute clipped ratio
        clip_eps = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

        # Compute policy loss
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Compute value loss (assuming rewards are advantages for simplicity)
        value_loss = (rewards - current_mean).pow(2).mean()

        # Compute entropy bonus
        entropy_bonus = - (current_log_prob * torch.exp(current_log_prob)).mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_bonus

        return loss

    def update_policy(self, loss: torch.Tensor):
        """
        ポリシーの更新を実施する。
        use_adam_optimizer が True の場合は Adam を用い、
        それ以外は Mirror Descent 更新を行う。
        """
        if self.use_adam_optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            raise NotImplementedError("Mirror Descent update is not implemented.")

    def adjust_hyperparameters(self, current_loss: float, previous_loss: float):
        """
        シンプルなルールに基づくオンラインハイパーパラメータ調整。
        将来的にはハイパーグラデント法等による自動調整への拡張も検討する。
        """
        if current_loss < previous_loss:
            self.eta *= 1.05
            self.alpha *= 1.05
        else:
            self.eta *= 0.9
            self.alpha *= 0.9

##############################################
# Training Loop Example
##############################################

if __name__ == '__main__':
    # 設定例
    state_dim = 128       # 入力状態次元
    action_dim = 2        # アクション空間次元
    num_samples_per_state = 8  # 1状態あたりの候補数

    # FastAttention の設定
    attention_config = FastAttentionConfig(
        d_model=128,     # FastAttention 出力次元
        d_key=32,
        d_query=32,
        n_heads=4,
        rank=16,
        rff_dim=64,
        k_max=32,
        stride=4,
        lsh_buckets=64,
        lsh_bandwidth=2.0,
        lsh_key_dim=32,
        wu_manber_prefix_len=4,
        dropout=0.1,
        intermediate_dim=256,
        use_rff=True
    )

    # SPPO エージェント初期化 (FastAttention 利用)
    sppo_agent = SPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        num_samples=num_samples_per_state,
        attention_config=attention_config,
        use_adam_optimizer=True
    )

    previous_loss = float('inf')
    num_episodes = 10
    num_steps_per_episode = 20

    # トレーニングループ
    for episode in range(num_episodes):
        for step in range(num_steps_per_episode):
            # ダミー状態生成 & 正規化
            dummy_states = torch.randn(num_samples_per_state, state_dim)
            normalized_states = sppo_agent.normalize_state(dummy_states)
            
            # アクションのサンプリング (再パラメータ化)
            actions, _, _ = sppo_agent.sample_actions(normalized_states)
            
            # 現実的報酬モデルを用いたグループ相対報酬計算
            group_rewards = sppo_agent.compute_group_rewards(normalized_states, actions, realistic_reward_model)
            
            # PPO 損失の計算とポリシー更新
            loss = sppo_agent.compute_ppo_loss(normalized_states, actions, group_rewards)
            sppo_agent.update_policy(loss)
            
            if step % 5 == 0:
                print(f"Episode: {episode}, Step: {step}, Loss: {loss.item():.4f}")
        
        # エピソード終了後にベースライン更新とハイパーパラメータ調整
        sppo_agent.update_baseline()
        current_loss = loss.item()
        sppo_agent.adjust_hyperparameters(current_loss, previous_loss)
        previous_loss = current_loss
        print(f"Episode {episode} completed, Adjusted eta: {sppo_agent.eta:.4f}, alpha: {sppo_agent.alpha:.4f}")
