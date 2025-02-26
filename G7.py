import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

##############################################
# Configuration
##############################################

@dataclass
class FastAttentionConfig:
    """Configuration for Fast Attention components."""
    d_model: int          # Model dimension
    d_key: int            # Key dimension
    d_query: int          # Query dimension
    n_heads: int          # Number of attention heads
    rank: int             # Low-rank approximation rank
    rff_dim: int          # Random Fourier features dimension
    k_max: int            # Maximum candidate keys per query
    stride: int           # Stride length for Trie
    lsh_buckets: int      # Number of LSH buckets
    lsh_bandwidth: float  # LSH bandwidth parameter
    lsh_key_dim: int      # LSH input dimension
    wu_manber_prefix_len: int  # Wu-Manber prefix length
    dropout: float = 0.1  # Dropout probability
    intermediate_dim: int = 2048  # FFN dimension
    use_rff: bool = True  # Use Random Fourier Features
    hyper_cuts_dim_groups: Optional[List[int]] = None  # Dimension groups
    n_lsh_hashes: int = 4  # Number of LSH hash functions

@dataclass
class TrainingConfig:
    """Configuration for training the RL agent."""
    state_dim: int        # Dimension of state space
    action_dim: int       # Dimension of action space
    hidden_dim: int = 256 # Hidden dimension for policy network
    policy_lr: float = 3e-4  # Learning rate for policy network
    eta: float = 0.01     # Initial step size parameter
    alpha: float = 0.1    # Initial trust region parameter
    num_samples: int = 64 # Number of samples per batch
    gamma: float = 0.99   # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    clip_eps: float = 0.2  # PPO clipping parameter
    entropy_coef: float = 0.01  # Entropy coefficient
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    use_attention: bool = True  # Whether to use attention
    seq_len: int = 8      # Sequence length for attention
    use_reward_normalization: bool = True  # Normalize rewards
    n_updates: int = 10   # Number of updates per batch
    total_timesteps: int = 1_000_000  # Total timesteps to train
    eval_frequency: int = 10000  # Evaluation frequency
    save_frequency: int = 50000  # Save frequency
    log_frequency: int = 1000  # Logging frequency
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use

##############################################
# Utility Functions
##############################################

def optimized_sqrt(n: int) -> float:
    """Optimized square root for powers of 2."""
    if n & (n - 1) == 0:  # Check if n is a power of 2
        k = n.bit_length() - 1
        return 2 ** (k / 2)
    return math.sqrt(n)

def create_logger(log_dir: str):
    """Create a simple logger."""
    import logging
    logger = logging.getLogger("RL_LOGGER")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Create file handler
    fh = logging.FileHandler(f"{log_dir}/training.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

##############################################
# Reward Model
##############################################

def realistic_reward_model(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    A realistic reward model for reinforcement learning environments that calculates
    rewards based on both state information and action selections.
    
    The model incorporates multiple reward components:
    1. Goal-oriented rewards: How close actions bring the agent to target states
    2. Energy efficiency: Penalizes excessive action magnitudes
    3. Smoothness: Rewards consistency and penalizes jittery behavior
    4. Exploration bonus: Rewards exploring under-visited state regions
    5. Task-specific rewards: Adjusts rewards based on domain-specific criteria
    
    Args:
        states: Tensor of shape [batch_size, state_dim] representing states
        actions: Tensor of shape [batch_size, action_dim] representing actions
    
    Returns:
        rewards: Tensor of shape [batch_size] containing computed rewards
    """
    batch_size = states.size(0)
    state_dim = states.size(1)
    action_dim = actions.size(1)
    device = states.device
    
    # Initialize reward components
    rewards = torch.zeros(batch_size, device=device)
    
    # 1. Goal-oriented rewards
    # Simulate target states - in a real environment, these would be defined externally
    # Here we create synthetic targets based on a simple function of the states
    target_direction = torch.ones(state_dim, device=device) / math.sqrt(state_dim)
    state_projections = torch.matmul(states, target_direction)
    
    # Calculate how well the actions align with moving toward target direction
    action_alignment = torch.zeros(batch_size, device=device)
    if action_dim >= 2:
        # For multi-dimensional actions, calculate alignment with target direction
        # Extract a subset of the state to determine the desired action direction
        state_subset = states[:, :min(state_dim, action_dim)]
        desired_action_direction = torch.nn.functional.normalize(state_subset, dim=1)
        normalized_actions = torch.nn.functional.normalize(actions, dim=1)
        # Cosine similarity between action and desired direction
        action_alignment = torch.sum(normalized_actions * desired_action_direction[:, :action_dim], dim=1)
    
    # Goal reward component - higher for actions that align with target direction
    goal_rewards = 2.0 * action_alignment + 0.5 * torch.tanh(state_projections)
    
    # 2. Energy efficiency rewards - penalize high-magnitude actions
    action_magnitude = torch.norm(actions, dim=1)
    energy_penalty = -0.1 * torch.square(action_magnitude)
    
    # 3. Smoothness rewards - simulate by comparing to "previous" actions
    # Since we don't have actual previous actions, we'll create a proxy using state information
    proxy_prev_actions = torch.tanh(states[:, :action_dim] * 0.1)
    action_diff = torch.norm(actions - proxy_prev_actions, dim=1)
    smoothness_reward = 0.2 * torch.exp(-2.0 * action_diff)
    
    # 4. Exploration bonus - reward exploring less-visited regions of state space
    # Simulate a density estimate using random projections of the state
    num_projections = 5
    projection_dims = min(10, state_dim)
    projections = torch.randn(num_projections, state_dim, projection_dims, device=device)
    projected_states = torch.stack([torch.matmul(states, proj) for proj in projections])
    
    # Quantize projected states to create "visitation buckets"
    quantized = torch.floor(projected_states * 2.0) / 2.0
    
    # Calculate "novelty" as an inverse function of an approximate density
    state_encodings = torch.cat([q.view(batch_size, -1) for q in quantized], dim=1)
    # Use self-similarity as a proxy for density
    similarities = torch.matmul(state_encodings, state_encodings.t())
    density_estimates = torch.sum(torch.exp(-0.1 * torch.abs(similarities)), dim=1) / batch_size
    exploration_bonus = 0.1 * (1.0 / (density_estimates + 0.1))
    
    # 5. Task-specific rewards
    # Simulate some task-specific reward conditions based on state features
    # For example, in control tasks, being close to center might be good
    state_balance = -0.2 * torch.mean(torch.abs(states), dim=1)  # Reward being close to origin
    
    # Add a penalty for actions that are too extreme in either direction
    boundary_penalty = -0.3 * torch.sum(torch.relu(torch.abs(actions) - 0.8), dim=1)
    
    # Combine all reward components
    rewards = (
        goal_rewards +
        energy_penalty +
        smoothness_reward +
        exploration_bonus +
        state_balance +
        boundary_penalty
    )
    
    # Add some small Gaussian noise to make rewards more realistic
    noise = 0.05 * torch.randn_like(rewards)
    rewards += noise
    
    # Optional: Additional reward shaping for specific states or actions
    # For instance, we could add extra rewards for actions that keep the agent balanced
    state_volatility = torch.std(states, dim=1)
    volatility_reward = 0.1 * torch.exp(-2.0 * state_volatility)
    rewards += volatility_reward
    
    return rewards

def batch_realistic_reward_model(states: torch.Tensor, actions: torch.Tensor, step_indices: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """
    Batched version of the realistic reward model that provides extended reward information.
    
    Args:
        states: Tensor of shape [batch_size, state_dim]
        actions: Tensor of shape [batch_size, action_dim]
        step_indices: Optional tensor of shape [batch_size] containing step indices for time-dependent rewards
    
    Returns:
        Dictionary containing:
            'total_rewards': Overall rewards tensor [batch_size]
            'goal_rewards': Goal-oriented reward component
            'energy_rewards': Energy efficiency component
            'smoothness_rewards': Action smoothness component
            'exploration_rewards': Exploration bonus component
            'task_rewards': Task-specific rewards
    """
    # Calculate basic rewards
    base_rewards = realistic_reward_model(states, actions)
    
    # For demonstration, break down into components
    batch_size = states.size(0)
    state_dim = states.size(1)
    action_dim = actions.size(1)
    device = states.device
    
    # Calculate each component separately for reporting
    # Goal-oriented rewards
    target_direction = torch.ones(state_dim, device=device) / math.sqrt(state_dim)
    state_projections = torch.matmul(states, target_direction)
    
    action_alignment = torch.zeros(batch_size, device=device)
    if action_dim >= 2:
        state_subset = states[:, :min(state_dim, action_dim)]
        desired_action_direction = torch.nn.functional.normalize(state_subset, dim=1)
        normalized_actions = torch.nn.functional.normalize(actions, dim=1)
        action_alignment = torch.sum(normalized_actions * desired_action_direction[:, :action_dim], dim=1)
    
    goal_rewards = 2.0 * action_alignment + 0.5 * torch.tanh(state_projections)
    
    # Energy efficiency
    action_magnitude = torch.norm(actions, dim=1)
    energy_rewards = -0.1 * torch.square(action_magnitude)
    
    # Smoothness
    proxy_prev_actions = torch.tanh(states[:, :action_dim] * 0.1)
    action_diff = torch.norm(actions - proxy_prev_actions, dim=1)
    smoothness_rewards = 0.2 * torch.exp(-2.0 * action_diff)
    
    # Exploration bonus (simplified for reporting)
    state_hash = torch.sum(torch.sin(states * 5.0), dim=1)
    exploration_rewards = 0.1 * torch.cos(state_hash * 3.0)
    
    # Task-specific rewards
    state_balance = -0.2 * torch.mean(torch.abs(states), dim=1)
    boundary_penalty = -0.3 * torch.sum(torch.relu(torch.abs(actions) - 0.8), dim=1)
    task_rewards = state_balance + boundary_penalty
    
    # Add time-dependent rewards if step indices are provided
    time_rewards = torch.zeros_like(base_rewards)
    if step_indices is not None:
        # Example: Rewards that decay over time to encourage quick task completion
        time_factor = torch.exp(-0.05 * step_indices.float())
        time_rewards = 0.5 * time_factor
    
    # Return all reward components
    return {
        'total_rewards': base_rewards,
        'goal_rewards': goal_rewards,
        'energy_rewards': energy_rewards,
        'smoothness_rewards': smoothness_rewards,
        'exploration_rewards': exploration_rewards,
        'task_rewards': task_rewards,
        'time_rewards': time_rewards
    }

##############################################
# Core Attention Components
##############################################

class LowRankLinear(nn.Module):
    """Low-rank approximation of a linear transformation."""
    
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.u_weight = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.v_weight = nn.Parameter(torch.randn(rank, out_features) / math.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(x, self.u_weight), self.v_weight)

class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for kernel approximation."""
    
    def __init__(self, input_dim: int, rff_dim: int):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi, requires_grad=False)
        self.scale = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = x.matmul(self.omega) + self.bias
        return torch.cos(projection) * self.scale

class LSHTable(nn.Module):
    """Locality-Sensitive Hashing for approximate nearest neighbor search."""
    
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

##############################################
# Candidate Search
##############################################

class CandidateFinder(nn.Module):
    """Finds candidate keys for queries using multiple search strategies."""
    
    def __init__(self, config: FastAttentionConfig, lsh_tables: nn.ModuleList):
        super().__init__()
        self.config = config
        self.lsh_tables = lsh_tables
        self.wu_manber_prefix_len = config.wu_manber_prefix_len
        self.hyper_cuts_dim_groups = config.hyper_cuts_dim_groups
        
    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert features to binary representation."""
        return (x > 0).float()
        
    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        """Split features along dimension groups."""
        if self.hyper_cuts_dim_groups is None:
            return [features]

        groups = []
        start = 0
        for group_dim in self.hyper_cuts_dim_groups:
            groups.append(features[:, :, start:start+group_dim])
            start += group_dim
        return groups
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, head_idx: int) -> torch.Tensor:
        """Find candidate keys for each query."""
        B, L, _ = query.size()
        device = query.device
        
        # Simplified candidate finding logic - using LSH
        query_bin = self.binary_quantize(query)
        key_bin = self.binary_quantize(key)
        
        # Use LSH to find approximate neighbors
        query_hash = self.lsh_tables[head_idx][0](query_bin.reshape(B*L, -1)).reshape(B, L, -1)
        key_hash = self.lsh_tables[head_idx][0](key_bin.reshape(B*L, -1)).reshape(B, L, -1)
        
        # Initialize candidates tensor
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=device)
        
        # For each batch and query position, find matching keys
        for b in range(B):
            for i in range(L):
                # Find keys with matching hash values
                matches = []
                for j in range(L):
                    # Simple hash matching (in real implementation, would use specialized data structures)
                    if torch.all(query_hash[b, i] == key_hash[b, j]):
                        matches.append(j)
                    if len(matches) >= self.config.k_max:
                        break
                        
                # Fill candidates with found matches
                if matches:
                    matches_tensor = torch.tensor(matches, dtype=torch.long, device=device)
                    candidates[b, i, :len(matches)] = matches_tensor[:self.config.k_max]
                    
        return candidates

##############################################
# Fast Attention Implementation
##############################################

class FastAttention(nn.Module):
    """Fast Attention using candidate search and low-rank projections."""
    
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        
        # Projections
        self.query_proj = nn.Linear(config.d_model, config.d_query * config.n_heads)
        self.key_proj = nn.Linear(config.d_model, config.d_key * config.n_heads)
        self.value_proj = nn.Linear(config.d_model, config.d_key * config.n_heads)
        
        # Value projections for each head
        self.value_up_projs = nn.ModuleList([
            LowRankLinear(config.d_key, config.d_model, config.rank)
            for _ in range(config.n_heads)
        ])
        
        # RFF encoders
        if config.use_rff:
            self.rff_encoders = nn.ModuleList([
                RandomFourierFeatures(config.d_key, config.rff_dim)
                for _ in range(config.n_heads)
            ])
            
        # LSH tables
        self.lsh_tables = nn.ModuleList([
            nn.ModuleList([
                LSHTable(config.lsh_key_dim, config.lsh_buckets, 
                         config.lsh_bandwidth, config.n_lsh_hashes)
            ])
            for _ in range(config.n_heads)
        ])
        
        # Candidate finder
        self.candidate_finder = CandidateFinder(config, self.lsh_tables)
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply fast attention mechanism."""
        B, L, _ = query.size()
        
        # Project inputs
        q = self.query_proj(query).view(B, L, self.config.n_heads, self.config.d_query)
        k = self.key_proj(key).view(B, L, self.config.n_heads, self.config.d_key)
        v = self.value_proj(value).view(B, L, self.config.n_heads, self.config.d_key)
        
        # Transpose for multi-head processing
        q = q.transpose(1, 2)  # [B, n_heads, L, d_query]
        k = k.transpose(1, 2)  # [B, n_heads, L, d_key]
        v = v.transpose(1, 2)  # [B, n_heads, L, d_key]
        
        head_outputs = []
        
        # Process each attention head
        for h in range(self.config.n_heads):
            # Find candidate keys
            candidates = self.candidate_finder(q[:, h], k[:, h], h)
            cand_mask = candidates != -1
            
            # Replace invalid indices with 0
            safe_candidates = candidates.clone()
            safe_candidates[safe_candidates == -1] = 0
            
            # Create batch indices for gathering
            b_idx = torch.arange(B, device=k.device).view(B, 1, 1).expand_as(candidates)
            
            # Gather candidate keys and values
            candidate_keys = k[:, h][b_idx, safe_candidates]
            candidate_values = v[:, h][b_idx, safe_candidates]
            
            # Apply RFF if enabled
            if self.config.use_rff:
                q_h = self.rff_encoders[h](q[:, h].reshape(-1, self.config.d_key))
                q_h = q_h.reshape(B, L, 1, self.config.rff_dim)
                
                candidate_keys = self.rff_encoders[h](candidate_keys.reshape(-1, self.config.d_key))
                candidate_keys = candidate_keys.reshape(B, L, candidates.size(-1), self.config.rff_dim)
                
                scale = optimized_sqrt(self.config.rff_dim)
            else:
                q_h = q[:, h].unsqueeze(2)  # [B, L, 1, d_key]
                scale = optimized_sqrt(self.config.d_key)
            
            # Compute similarity scores
            sim = torch.matmul(q_h, candidate_keys.transpose(-2, -1)).squeeze(2) / scale
            
            # Mask invalid candidates
            sim = sim.masked_fill(~cand_mask, float('-inf'))
            
            # Apply attention weights
            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Project values to output dimension
            candidate_values = self.value_up_projs[h](candidate_values.reshape(-1, self.config.d_key))
            candidate_values = candidate_values.reshape(B, L, candidates.size(-1), self.config.d_model)
            
            # Apply weights to values
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
    """Feed-Forward Network with two layers and ReLU activation."""
    
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

##############################################
# Encoder Layer
##############################################

class FastAttentionEncoderLayer(nn.Module):
    """Encoder layer with Fast Attention and feed-forward network."""
    
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.self_attn = FastAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        residual = src
        src = self.norm1(src)
        attn = self.self_attn(src, src, src, mask=src_mask)
        src = residual + self.dropout(attn)
        
        # Feed-forward with residual connection
        residual = src
        src = self.norm2(src)
        ffn = self.feed_forward(src)
        return residual + self.dropout(ffn)

##############################################
# Policy Network
##############################################

class PolicyNetworkWithAttention(nn.Module):
    """Policy network that integrates FastAttention for state processing."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 attention_config: Optional[FastAttentionConfig] = None,
                 seq_len: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.attention = FastAttentionEncoderLayer(attention_config) if attention_config else None
        self.effective_state_dim = attention_config.d_model if self.attention else state_dim
        
        # State encoder if needed to match attention dimensions
        if attention_config and state_dim != attention_config.d_model:
            self.state_encoder = nn.Linear(state_dim, attention_config.d_model)
        else:
            self.state_encoder = None
        
        # MLP for policy outputs
        self.net = nn.Sequential(
            nn.Linear(self.effective_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)  # mean, log_std
        )
        
        # Value network head
        self.value_net = nn.Sequential(
            nn.Linear(self.effective_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both policy and value network.
        
        Args:
            state: State tensor of shape [batch_size, state_dim] or [batch_size, seq_len, state_dim]
            
        Returns:
            mean: Action mean
            std: Action standard deviation
            value: State value estimate
        """
        if self.attention:
            # Process state as sequence
            if len(state.shape) == 2:
                batch_size = state.size(0)
                state = state.view(batch_size // self.seq_len, self.seq_len, -1)
                
            # Encode state if needed
            if self.state_encoder:
                state = self.state_encoder(state)
                
            # Apply attention
            state_features = self.attention(state)
            state_features = state_features.mean(dim=1)  # [batch, d_model]
        else:
            # Use raw state features
            if len(state.shape) == 3:
                state_features = state.mean(dim=1)
            else:
                state_features = state
        
        # Policy head
        policy_out = self.net(state_features)
        mean, log_std = torch.chunk(policy_out, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # Limit std range for stability
        std = log_std.exp()
        
        # Value head
        value = self.value_net(state_features)
        
        return mean, std, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Sample action from the policy distribution.
        
        Args:
            state: State tensor
            deterministic: Whether to return deterministic action (mean) or stochastic
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            info: Dictionary with additional info (mean, std, entropy, value)
        """
        with torch.no_grad():
            mean, std, value = self.forward(state)
            
            if deterministic:
                action = mean
                normal_dist = Normal(mean, std)
                log_prob = normal_dist.log_prob(action).sum(dim=-1)
            else:
                normal_dist = Normal(mean, std)
                action = normal_dist.rsample()  # Reparameterization trick
                log_prob = normal_dist.log_prob(action).sum(dim=-1)
                
            entropy = normal_dist.entropy().sum(dim=-1)
            
            info = {
                "mean": mean,
                "std": std,
                "value": value.squeeze(-1),
                "entropy": entropy
            }
            
            return action, log_prob, info

##############################################
# Memory Buffer
##############################################

class RolloutBuffer:
    """Buffer to store trajectories collected during training."""
    
    def __init__(self, config: TrainingConfig):
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.seq_len = config.seq_len
        self.max_size = config.num_samples
        self.device = config.device
        
        # Initialize buffers
        self.reset()
        
    def reset(self):
        """Reset buffer."""
        self.states = []
        self.actions = []
        self.rewards = []　# Very important: consider consolidation.
        self.


##############################################
# Policy Network　# ReCall：Very important: consider consolidation.
##############################################

class PolicyNetworkWithAttention(nn.Module):
    """Policy network that integrates FastAttention for state processing."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 attention_config: Optional[FastAttentionConfig] = None,
                 seq_len: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.attention = FastAttentionEncoderLayer(attention_config) if attention_config else None
        self.effective_state_dim = attention_config.d_model if self.attention else state_dim
        
        # MLP for policy outputs
        self.net = nn.Sequential(
            nn.Linear(self.effective_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)  # mean, log_std
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attention:
            # Process state as sequence
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
# SPPO Agent
##############################################

class SPPO:
    """SPPO Agent with FastAttention integration for state processing."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy_lr: float = 1e-3,
        eta: float = 0.01,
        alpha: float = 0.1,
        num_samples: int = 8,
        use_adam_optimizer: bool = True,
        attention_config: Optional[FastAttentionConfig] = None,
        seq_len: int = 1
    ):
        # Initialize policy networks
        self.policy = PolicyNetworkWithAttention(
            state_dim, action_dim, attention_config=attention_config, seq_len=seq_len)
        self.baseline_policy = PolicyNetworkWithAttention(
            state_dim, action_dim, attention_config=attention_config, seq_len=seq_len)
        self.baseline_policy.load_state_dict(self.policy.state_dict())
        
        # Hyperparameters
        self.eta = eta
        self.alpha = alpha
        self.num_samples = num_samples
        
        # Optimizer
        if use_adam_optimizer:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        else:
            self.optimizer = None
            
    def normalize_state(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize state tensor to zero mean and unit variance."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + eps)
        
    def sample_actions(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions using reparameterization trick."""
        mean, std = self.policy(state)
        epsilon = torch.randn_like(mean)
        actions = mean + std * epsilon
        return actions, mean, std
        
    def compute_log_prob(self, actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions in Gaussian distribution."""
        return (-0.5 * ((actions - mean) / std).pow(2) - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
        
    def compute_group_rewards(self, state: torch.Tensor, actions: torch.Tensor, 
                             reward_model: Callable) -> torch.Tensor:
        """Compute normalized rewards within a group."""
        raw_rewards = reward_model(state, actions)
        mean_reward = raw_rewards.mean()
        std_reward = raw_rewards.std() + 1e-8
        normalized_rewards = (raw_rewards - mean_reward) / std_reward
        return normalized_rewards.unsqueeze(1)
        
    def update_baseline(self):
        """Update baseline policy with current policy parameters."""
        self.baseline_policy.load_state_dict(self.policy.state_dict())
        
    def compute_ppo_loss(self, state: torch.Tensor, actions: torch.Tensor, 
                         rewards: torch.Tensor, advantages: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute PPO loss with clipped objective."""
        if advantages is None:
            advantages = rewards
            
        # Current policy log prob
        current_mean, current_std = self.policy(state)
        current_log_prob = self.compute_log_prob(actions, current_mean, current_std)
        
        # Baseline policy log prob
        with torch.no_grad():
            baseline_mean, baseline_std = self.baseline_policy(state)
            baseline_log_prob = self.compute_log_prob(actions, baseline_mean, baseline_std)
        
        # Compute ratio and clipped objective
        ratio = torch.exp(current_log_prob - baseline_log_prob)
        clip_eps = 0.2
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        
        # Policy loss
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
        value_loss = (rewards - current_mean).pow(2).mean()
        
        # Entropy bonus
        entropy_bonus = -(current_log_prob * torch.exp(current_log_prob)).mean()
        
        # Total loss
        return policy_loss + 0.5 * value_loss + 0.01 * entropy_bonus
        
    def update_policy(self, loss: torch.Tensor):
        """Update policy parameters using optimizer."""
        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
    def adjust_hyperparameters(self, current_loss: float, previous_loss: float):
        """Adjust hyperparameters based on loss trends."""
        if current_loss < previous_loss:
            self.eta *= 1.05
            self.alpha *= 1.05
        else:
            self.eta *= 0.9
            self.alpha *= 0.9

##############################################
# Example Usage
##############################################

def example_training():
    """Example training loop with dummy reward model."""
    # Dummy reward model
    def dummy_reward_model(states, actions):
        return torch.sum(actions**2, dim=-1)
    
    # Configuration
    state_dim = 128
    action_dim = 2
    num_samples = 8
    
    # Fast Attention config
    attention_config = FastAttentionConfig(
        d_model=128,
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
    
    # Initialize agent
    agent = SPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        num_samples=num_samples,
        attention_config=attention_config,
        use_adam_optimizer=True
    )
    
    # Training loop
    previous_loss = float('inf')
    num_episodes = 3
    num_steps_per_episode = 5
    
    for episode in range(num_episodes):
        for step in range(num_steps_per_episode):
            # Generate dummy states
            states = torch.randn(num_samples, state_dim)
            norm_states = agent.normalize_state(states)
            
            # Sample actions
            actions, _, _ = agent.sample_actions(norm_states)
            
            # Compute rewards
            rewards = agent.compute_group_rewards(norm_states, actions, dummy_reward_model)
            
            # Update policy
            loss = agent.compute_ppo_loss(norm_states, actions, rewards)
            agent.update_policy(loss)
            
            print(f"Episode: {episode}, Step: {step}, Loss: {loss.item():.4f}")
            
        # Update baseline and adjust hyperparameters
        agent.update_baseline()
        current_loss = loss.item()
        agent.adjust_hyperparameters(current_loss, previous_loss)
        previous_loss = current_loss
        print(f"Episode {episode} completed, Adjusted eta: {agent.eta:.4f}, alpha: {agent.alpha:.4f}")

if __name__ == '__main__':
    example_training()
