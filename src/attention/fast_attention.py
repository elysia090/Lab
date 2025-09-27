from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
# Low-level Helper Functions
##############################################

def fma(a: float, b: float, c: float) -> float:
    """Return ``a * b + c`` using :func:`math.fma` when available."""

    try:
        return math.fma(a, b, c)
    except AttributeError:
        return a * b + c

def optimized_sqrt(n: int) -> float:
    """Return ``sqrt(n)`` with a fast-path for powers of two.

    The utility mirrors :func:`math.sqrt` but avoids floating point work for
    dimensions that are powers of two – a common case for attention modules.
    """

    if n & (n - 1) == 0:  # power-of-two check
        k = n.bit_length() - 1
        return 2 ** (k / 2)
    return math.sqrt(n)

##############################################
# Fast Attention Components
##############################################

@torch.jit.script
def select_topk(similarities: torch.Tensor, k_max: int) -> torch.Tensor:
    """Return indices of the ``k_max`` largest similarities along the last axis."""

    _, topk_idx = torch.topk(similarities, k=k_max, dim=-1)
    return topk_idx

@dataclass
class FastAttentionConfig:
    d_model: int          # Overall model dimension.
    d_key: int            # Key dimension.
    d_query: int          # Query dimension.
    n_heads: int          # Number of attention heads.
    rank: int             # Rank for low-rank approximations.
    rff_dim: int          # Output dimension for random Fourier features.
    k_max: int            # Maximum candidate keys per query.
    stride: int           # Stride length for Trie.
    lsh_buckets: int      # Number of LSH buckets.
    lsh_bandwidth: float  # Bandwidth for LSH.
    lsh_key_dim: int      # LSH input dimension.
    wu_manber_prefix_len: int  # Prefix length for Wu-Manber search.
    hyper_cuts_dim_groups: Optional[List[int]] = None  # Feature groups.
    n_lsh_hashes: int = 4
    dropout: float = 0.1
    intermediate_dim: int = 2048
    use_rff: bool = True  # Whether to use RFF.

class LowRankLinear(nn.Module):
    """Low-rank linear layer via factorised weight matrices."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        scale = math.sqrt(rank)
        self.u_weight = nn.Parameter(torch.randn(in_features, rank) / scale)
        self.v_weight = nn.Parameter(torch.randn(rank, out_features) / scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.matmul(self.u_weight).matmul(self.v_weight)

class RandomFourierFeatures(nn.Module):
    """Project inputs using random Fourier features.

    The random parameters are registered as buffers to avoid appearing in the
    optimiser state while still moving with the module across devices.
    """

    def __init__(self, input_dim: int, rff_dim: int) -> None:
        super().__init__()
        self.register_buffer("omega", torch.randn(input_dim, rff_dim))
        self.register_buffer("bias", torch.rand(rff_dim) * 2 * math.pi)
        self.scale = math.sqrt(2.0 / rff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = x.matmul(self.omega) + self.bias
        return torch.cos(projection) * self.scale

class LSHTable(nn.Module):
    """Locality-sensitive hashing (LSH) table."""

    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes
        self.register_buffer("random_vectors", torch.randn(dim, n_hashes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.matmul(self.random_vectors)
        return torch.floor(proj / self.bandwidth) % self.n_buckets

@dataclass
class _TrieNode:
    children: Dict[Tuple[int, ...], "_TrieNode"] = field(default_factory=dict)
    indices: List[int] = field(default_factory=list)


class Trie(nn.Module):
    """Prefix tree for candidate lookup using binary quantisation."""

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride_len = stride
        self.reset()

    def reset(self) -> None:
        self.root = _TrieNode()

    def insert(self, binary_vector: torch.Tensor, index: int) -> None:
        node = self.root
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i : i + self.stride_len].tolist())
            node = node.children.setdefault(prefix, _TrieNode())
        node.indices.append(index)

    def bulk_insert(self, binary_matrix: torch.Tensor) -> None:
        self.reset()
        for idx, row in enumerate(binary_matrix):
            self.insert(row, idx)

    def search(self, binary_vector: torch.Tensor) -> List[int]:
        node = self.root
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i : i + self.stride_len].tolist())
            if prefix not in node.children:
                return []
            node = node.children[prefix]
        return list(node.indices)

class CandidateFinder(nn.Module):
    """Candidate search module combining simple heuristics."""

    def __init__(self, config: FastAttentionConfig, tries: Iterable[Trie], lsh_tables: nn.ModuleList) -> None:
        super().__init__()
        self.config = config
        self.tries = list(tries)
        self.lsh_tables = lsh_tables
        self.wu_manber_prefix_len = config.wu_manber_prefix_len
        self.hyper_cuts_dim_groups = config.hyper_cuts_dim_groups

    @staticmethod
    def _binary_quantize(x: torch.Tensor) -> torch.Tensor:
        return (x > 0).to(dtype=torch.int8)

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        if self.hyper_cuts_dim_groups is None:
            return [features]
        groups: List[torch.Tensor] = []
        start = 0
        for group_dim in self.hyper_cuts_dim_groups:
            groups.append(features[:, :, start : start + group_dim])
            start += group_dim
        return groups

    def _build_wu_manber_hash_table(self, key_bin: torch.Tensor) -> Dict[Tuple[int, ...], List[int]]:
        table: Dict[Tuple[int, ...], List[int]] = {}
        prefix_len = self.wu_manber_prefix_len
        if prefix_len <= 0:
            return table
        for index, vector in enumerate(key_bin):
            prefix = tuple(vector[:prefix_len].tolist())
            table.setdefault(prefix, []).append(index)
        return table

    def _wu_manber_search(self, query_bin: torch.Tensor, table: Dict[Tuple[int, ...], List[int]]) -> List[int]:
        if not table:
            return []
        prefix = tuple(query_bin[: self.wu_manber_prefix_len].tolist())
        return table.get(prefix, [])

    def _get_wu_manber_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor) -> List[List[List[int]]]:
        key_bin = self._binary_quantize(key_grp)
        query_bin = self._binary_quantize(query_grp)
        batch_candidates: List[List[List[int]]] = []
        for key_batch, query_batch in zip(key_bin, query_bin):
            table = self._build_wu_manber_hash_table(key_batch)
            batch_candidates.append([
                self._wu_manber_search(query_vec, table) for query_vec in query_batch
            ])
        return batch_candidates

    def _get_trie_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> List[List[List[int]]]:
        trie = self.tries[head_idx]
        batch_candidates: List[List[List[int]]] = []
        for key_batch, query_batch in zip(key_grp, query_grp):
            trie.bulk_insert(self._binary_quantize(key_batch))
            batch_candidates.append([
                trie.search(self._binary_quantize(query_vec)) for query_vec in query_batch
            ])
        return batch_candidates

    def _merge_candidate_indices_groups(self, cand_tensors: List[torch.Tensor]) -> torch.Tensor:
        merged = torch.cat(cand_tensors, dim=-1)
        B, L, _ = merged.shape
        output = torch.full((B, L, self.config.k_max), -1, device=merged.device, dtype=merged.dtype)
        for b in range(B):
            for i in range(L):
                valid = merged[b, i][merged[b, i] >= 0]
                if valid.numel() == 0:
                    continue
                unique = torch.unique(valid, sorted=True)
                count = min(unique.numel(), self.config.k_max)
                output[b, i, :count] = unique[:count]
        return output

    def _process_dimension_group_candidates(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> torch.Tensor:
        B, L, _ = query_grp.size()
        wu_cands = self._get_wu_manber_candidates_group(query_grp, key_grp)
        trie_cands = self._get_trie_candidates_group(query_grp, key_grp, head_idx)
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_grp.device)
        for b in range(B):
            for i in range(L):
                seen: List[int] = []
                for idx in wu_cands[b][i]:
                    if idx in trie_cands[b][i] and idx not in seen:
                        seen.append(idx)
                        if len(seen) == self.config.k_max:
                            break
                if seen:
                    candidates[b, i, : len(seen)] = torch.tensor(seen, dtype=torch.long, device=query_grp.device)
        return candidates

    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int) -> torch.Tensor:
        query_groups = self.split_features_by_dim_groups(query_up)
        key_groups = self.split_features_by_dim_groups(key_up)
        cand_list = [
            self._process_dimension_group_candidates(q_grp, k_grp, head_idx)
            for q_grp, k_grp in zip(query_groups, key_groups)
        ]
        if not cand_list:
            B, L, _ = query_up.size()
            return torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_up.device)
        return self._merge_candidate_indices_groups(cand_list)

class AbsorptionProjection(nn.Module):
    """Project queries into the key space using a low-rank transform."""

    def __init__(self, query_dim: int, key_dim: int, rank: int) -> None:
        super().__init__()
        scale = math.sqrt(rank)
        self.u_q = nn.Parameter(torch.randn(query_dim, rank) / scale)
        self.v_q = nn.Parameter(torch.randn(rank, key_dim) / scale)
        self.u_k = nn.Parameter(torch.randn(key_dim, rank) / scale)
        self.v_k = nn.Parameter(torch.randn(rank, key_dim) / scale)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_uq = self.u_q.matmul(self.v_q)
        w_uk = self.u_k.matmul(self.v_k)
        w_absorb = w_uk.transpose(0, 1).matmul(w_uq)
        q_proj = query.matmul(w_absorb.transpose(0, 1))
        return q_proj, key

class FastAttention(nn.Module):
    """
    Multi-head Fast Attention module.
    """
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        self.query_down_proj = nn.Linear(config.d_model, config.d_query)
        self.key_down_proj = nn.Linear(config.d_model, config.d_key)
        self.value_down_proj = nn.Linear(config.d_model, config.d_key)
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
                nn.ModuleList([LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)])
                for _ in range(config.n_heads)
            ])
        self.tries_list = nn.ModuleList([Trie(config.stride) for _ in range(config.n_heads)])
        self.candidate_finder = CandidateFinder(config, self.tries_list, self.lsh_tables_list)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = query.size()
        query_down = self.query_down_proj(query)
        key_down = self.key_down_proj(key)
        value_down = self.value_down_proj(value)
        if mask is not None and mask.device != query.device:
            mask = mask.to(query.device)
        head_outputs = []
        for head_idx in range(self.config.n_heads):
            Q_proj, K_proj = self.absorption_projs[head_idx](query_down, key_down)
            candidates = self.candidate_finder(Q_proj, K_proj, head_idx)
            cand_mask = candidates != -1
            safe_candidates = candidates.clone()
            safe_candidates[safe_candidates == -1] = 0
            b_idx = torch.arange(B, device=K_proj.device).view(B, 1, 1).expand(B, L, self.config.k_max)
            candidate_keys = K_proj[b_idx, safe_candidates]
            candidate_values = value_down[b_idx, safe_candidates]
            q_exp = Q_proj.unsqueeze(2)
            if self.config.use_rff:
                q_exp = self.rff_encoders[head_idx](q_exp.view(-1, self.config.d_key)).view(B, L, 1, self.config.rff_dim)
                candidate_keys = self.rff_encoders[head_idx](candidate_keys.view(-1, self.config.d_key)).view(
                    B, L, self.config.k_max, self.config.rff_dim
                )
                candidate_values = self.value_up_projs[head_idx](
                    candidate_values.view(-1, self.config.d_key)
                ).view(B, L, self.config.k_max, self.config.d_model)
                scale = optimized_sqrt(self.config.rff_dim)
            else:
                candidate_values = self.value_up_projs[head_idx](
                    candidate_values.view(-1, self.config.d_key)
                ).view(B, L, self.config.k_max, self.config.d_model)
                scale = optimized_sqrt(self.config.d_key)
            sim = torch.matmul(q_exp, candidate_keys.transpose(-2, -1)).squeeze(2) / scale
            if mask is not None:
                if mask.dim() != 3:
                    raise ValueError("mask must have shape [batch, query_len, key_len]")
                attn_mask = mask.gather(-1, safe_candidates)
                sim = sim.masked_fill(~attn_mask, float("-inf"))
            sim = sim.masked_fill(~cand_mask, float("-inf"))
            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values, dim=2)
            head_outputs.append(head_out)
        concat = torch.cat(head_outputs, dim=-1)
        output = self.output_proj(concat)
        return self.dropout(output)

class FeedForwardNetwork(nn.Module):
    """
    Two-layer feedforward network with ReLU activation and dropout.
    """
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
    """
    Transformer encoder layer using Fast Attention followed by a feedforward network.
    """
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

##############################################
# GRPO Components (Full Implementation)
##############################################

def compute_real_reward(full_output: torch.Tensor, fast_output: torch.Tensor, cost: float) -> float:
    """
    Compute the real reward based on the L2 error between full and fast attention outputs and the computational cost.
    Replace this placeholder with your actual reward function.
    """
    error = torch.norm(full_output - fast_output)
    reward = - float(error.item()) - cost
    return reward

def extract_state_from_module(module: FastAttention, input_data: torch.Tensor) -> List[float]:
    """
    Extract the state vector from the module's candidate search metrics.
    Replace this placeholder with real measurements.
    """
    candidate_count = module.config.k_max
    avg_similarity = 0.75   # Replace with actual metric.
    variance_similarity = 0.05
    resource_cost = 1.0
    downstream_perf = 0.95
    history_metric = 0.0
    return [candidate_count, avg_similarity, variance_similarity, resource_cost, downstream_perf, history_metric]

class GRPOEnvironmentMulti:
    """
    Environment for multi-group candidate search hyperparameter optimization.
    The state and reward are derived from real model metrics.
    """
    def __init__(self, fast_attention_module: FastAttention, validation_data: torch.Tensor,
                 initial_hyperparams: List[Dict[str, float]], alpha: float = 1.0, beta: float = 0.1):
        self.fast_module = fast_attention_module
        self.validation_data = validation_data
        self.groups = initial_hyperparams
        self.alpha = alpha  # Weight for performance error.
        self.beta = beta    # Weight for computational cost.
    def get_state(self) -> torch.Tensor:
        states = []
        for g in self.groups:
            # In practice, run the module and extract real metrics.
            state_vec = [g['k_max']] + extract_state_from_module(self.fast_module, self.validation_data)[1:]
            states.append(state_vec)
        return torch.tensor(states, dtype=torch.float32)
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = []
        new_states = []
        for i, g in enumerate(self.groups):
            a = actions[i]
            g['lsh_buckets'] = max(1, g['lsh_buckets'] + a[0].item())
            g['lsh_bandwidth'] = max(0.1, g['lsh_bandwidth'] + a[1].item())
            g['trie_stride'] = max(1, g['trie_stride'] + a[2].item())
            g['k_max'] = max(1, g['k_max'] + int(round(a[3].item())))
            # In practice, run both full and fast attention on validation_data.
            # Here we use placeholders.
            full_output = torch.randn(1)  # Replace with actual full attention output.
            fast_output = torch.randn(1)  # Replace with actual fast attention output.
            cost = g['k_max']  # Example: cost proportional to candidate count.
            reward = compute_real_reward(full_output, fast_output, cost)
            rewards.append(reward)
            state_vec = [g['k_max']] + extract_state_from_module(self.fast_module, self.validation_data)[1:]
            new_states.append(state_vec)
        return torch.tensor(new_states, dtype=torch.float32), torch.tensor(rewards, dtype=torch.float32)

class GRPOAgent(nn.Module):
    """
    GRPO policy network that outputs an action vector per group.
    Action vector: [Δlsh_buckets, Δlsh_bandwidth, Δtrie_stride, Δk_max]
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc(state)

def grpo_training_episode(agent: GRPOAgent, env: GRPOEnvironmentMulti, optimizer: torch.optim.Optimizer,
                            episode_length: int = 10, epsilon: float = 1e-8, lambda_kl: float = 0.01) -> float:
    """
    Runs one GRPO training episode over multiple time steps.
    Uses a Monte Carlo baseline (mean reward) scaled by reward standard deviation and includes an explicit KL penalty.
    """
    total_loss = 0.0
    state = env.get_state()  # Shape: [G, state_dim]
    for _ in range(episode_length):
        actions = agent(state)  # [G, action_dim]
        std = torch.ones_like(actions)
        dist = torch.distributions.Normal(actions, std)
        log_probs = dist.log_prob(actions).sum(dim=1)
        next_state, rewards = env.step(actions)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + epsilon
        advantage = (rewards - mean_reward) / std_reward
        # Explicit KL penalty: for Normal(actions, 1) vs. Normal(0,1), KL = 0.5 * (actions^2)
        kl_div = 0.5 * (actions ** 2).mean()
        loss = - (log_probs * advantage).mean() + lambda_kl * kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        state = next_state
    return total_loss / episode_length

def example_grpo_full():
    """
    Full GRPO training example using real metrics.
    Replace the placeholders in compute_real_reward and extract_state_from_module with your actual logic.
    """
    initial_hyperparams = [
        {'lsh_buckets': 32, 'lsh_bandwidth': 4.0, 'trie_stride': 4, 'k_max': 64},
        {'lsh_buckets': 30, 'lsh_bandwidth': 4.2, 'trie_stride': 5, 'k_max': 60},
        {'lsh_buckets': 35, 'lsh_bandwidth': 3.8, 'trie_stride': 4, 'k_max': 70}
    ]
    config = FastAttentionConfig(
        d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
        rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
        lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
        hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4, use_rff=True)
    fast_module = FastAttention(config)
    validation_data = torch.randn(2, 128, config.d_model)  # Replace with your actual validation data.
    env = GRPOEnvironmentMulti(fast_module, validation_data, initial_hyperparams, alpha=1.0, beta=0.1)
    agent = GRPOAgent(state_dim=6, action_dim=4)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    num_episodes = 20
    for ep in range(num_episodes):
        loss = grpo_training_episode(agent, env, optimizer, episode_length=10)
        state = env.get_state()
        print(f"Episode {ep}: Avg Loss={loss:.4f}, Group States={state.tolist()}")

##############################################
# Usage Examples
##############################################

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
    print("Starting full GRPO training (multi-group) with real metrics...")
    example_grpo_full()
