import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Callable
from torch.utils.tensorboard import SummaryWriter

##############################################
# Low-level Helpers 
##############################################

def optimized_sqrt(n: int) -> float:
    if n & (n - 1) == 0:
        k = n.bit_length() - 1
        return 2 ** (k / 2)
    return math.sqrt(n)

def fma(a: float, b: float, c: float) -> float:
    try:
        return math.fma(a, b, c)
    except AttributeError:
        return a * b + c

##############################################
# Fast Attention Components 
##############################################

@dataclass
class FastAttentionConfig:
    d_model: int; d_key: int; d_query: int; n_heads: int; rank: int; rff_dim: int; k_max: int; stride: int
    lsh_buckets: int; lsh_bandwidth: float; lsh_key_dim: int; wu_manber_prefix_len: int
    hyper_cuts_dim_groups: Optional[List[int]] = None; n_lsh_hashes: int = 4; dropout: float = 0.1
    intermediate_dim: int = 2048; use_rff: bool = True

class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.u_weight = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
        self.v_weight = nn.Parameter(torch.randn(rank, out_features) / math.sqrt(rank))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.u_weight).matmul(self.v_weight)

class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim: int, rff_dim: int):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(input_dim, rff_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.rand(rff_dim) * 2 * math.pi, requires_grad=False)
        self.scale = math.sqrt(2.0 / rff_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = x.matmul(self.omega) + self.bias
        return torch.cos(projection) * self.scale

class LSHTable(nn.Module):
    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes
        self.register_buffer("random_vectors", torch.randn(dim, n_hashes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x.matmul(self.random_vectors)
        return torch.floor(proj / self.bandwidth) % self.n_buckets

_TRIE_INDICES_KEY = '_indices'
class Trie(nn.Module):
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
        current_node.setdefault(_TRIE_INDICES_KEY, []).append(index)
    def search(self, binary_vector: torch.Tensor) -> List[int]:
        current_node = self.root_node
        for i in range(0, len(binary_vector), self.stride_len):
            prefix = tuple(binary_vector[i:i+self.stride_len].tolist())
            if prefix not in current_node:
                return []
            current_node = current_node[prefix]
        return current_node.get(_TRIE_INDICES_KEY, [])

class CandidateFinder(nn.Module):
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
        if self.hyper_cuts_dim_groups is None:
            return [features]
        groups = []
        start = 0
        for group_dim in self.hyper_cuts_dim_groups:
            groups.append(features[:, :, start:start+group_dim])
            start += group_dim
        return groups
    def _build_wu_manber_hash_table(self, key_bin: torch.Tensor) -> Dict[tuple, List[int]]:
        table: Dict[tuple, List[int]] = {}
        L = key_bin.size(0)
        for i in range(L):
            prefix = tuple(key_bin[i, :self.config.wu_manber_prefix_len].tolist())
            table.setdefault(prefix, []).append(i)
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
        for b in range(B):
            trie = Trie(self.config.stride)
            key_bin = self.binary_quantize(key_grp[b])
            for i in range(L):
                trie.insert(key_bin[i], i)
            batch_list = [trie.search(self.binary_quantize(query_grp[b][i])) for i in range(L)]
            cand_lists.append(batch_list)
        return cand_lists
    def merge_candidate_indices_groups(self, cand_tensors: List[torch.Tensor]) -> torch.Tensor:
        merged = torch.cat(cand_tensors, dim=-1)
        merged, _ = torch.sort(merged)
        return torch.unique_consecutive(merged, dim=-1)
    def _process_dimension_group_candidates(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> torch.Tensor:
        B, L, _ = query_grp.size()
        wu_cands = self._get_wu_manber_candidates_group(query_grp, key_grp)
        trie_cands = self._get_trie_candidates_group(query_grp, key_grp, head_idx)
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_grp.device)
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
        B, L, _ = query_up.size()
        query_groups = self.split_features_by_dim_groups(query_up)
        key_groups = self.split_features_by_dim_groups(key_up)
        cand_list = [self._process_dimension_group_candidates(q_grp, k_grp, head_idx)
                     for q_grp, k_grp in zip(query_groups, key_groups)]
        if cand_list:
            merged = self.merge_candidate_indices_groups(cand_list)
            return merged[:, :, :self.config.k_max]
        return torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_up.device)

class AbsorptionProjection(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, rank: int):
        super().__init__()
        self.u_q = nn.Parameter(torch.randn(query_dim, rank) / math.sqrt(rank))
        self.v_q = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))
        self.u_k = nn.Parameter(torch.randn(key_dim, rank) / math.sqrt(rank))
        self.v_k = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        W_UQ = torch.matmul(self.u_q, self.v_q)
        W_UK = torch.matmul(self.u_k, self.v_k)
        W_absorb = torch.matmul(W_UK.transpose(0, 1), W_UQ)
        Q_proj = torch.matmul(query, W_absorb.transpose(0, 1))
        return Q_proj, key

class FastAttention(nn.Module):
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.config = config
        self.query_down_proj = nn.Linear(config.d_model, config.d_query)
        self.key_value_down_proj = nn.Linear(config.d_model, config.d_key)
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
        self.candidate_finder = CandidateFinder(config, list(self.tries_list), self.lsh_tables_list)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = query.size()
        query_down = self.query_down_proj(query)
        key_down = self.key_value_down_proj(key)
        head_outputs = []
        for head_idx in range(self.config.n_heads):
            Q_proj, K_proj = self.absorption_projs[head_idx](query_down, key_down)
            candidates = self.candidate_finder(Q_proj, K_proj, head_idx)
            cand_mask = candidates != -1
            safe_candidates = candidates.clone()
            safe_candidates[safe_candidates == -1] = 0
            num_candidates = candidates.size(-1)
            b_idx = torch.arange(B, device=K_proj.device).view(B, 1, 1).expand(B, L, num_candidates)
            candidate_keys = K_proj[b_idx, safe_candidates]
            q_exp = Q_proj.unsqueeze(2)
            if self.config.use_rff:
                q_exp = self.rff_encoders[head_idx](q_exp.reshape(-1, self.config.d_key)).reshape(B, L, 1, self.config.rff_dim)
                candidate_keys = self.rff_encoders[head_idx](candidate_keys.reshape(-1, self.config.d_key)).reshape(B, L, num_candidates, self.config.rff_dim)
                scale = optimized_sqrt(self.config.rff_dim)
            else:
                scale = optimized_sqrt(self.config.d_key)
            sim = torch.matmul(q_exp, candidate_keys.transpose(-2, -1)).squeeze(2) / scale
            sim = sim.masked_fill(~cand_mask, float('-inf'))
            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)
            candidate_values = key_down[b_idx, safe_candidates]
            candidate_values = self.value_up_projs[head_idx](candidate_values.reshape(-1, self.config.d_key)).reshape(B, L, num_candidates, self.config.d_model)
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values, dim=2)
            head_outputs.append(head_out)
        concat = torch.cat(head_outputs, dim=-1)
        output = self.output_proj(concat)
        return self.dropout(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.intermediate_dim)
        self.linear2 = nn.Linear(config.intermediate_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class FastAttentionEncoderLayer(nn.Module):
    def __init__(self, config: FastAttentionConfig):
        super().__init__()
        self.self_attn = FastAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.norm1(src)
        attn = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout(attn)

        src = self.norm2(src)
        ffn = self.feed_forward(src)
        return src + self.dropout(ffn)

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
    def __init__(self, policy_network: FastAttentionEncoderLayer, k_responses: int):
        self.policy_network = policy_network
        self.k_responses = k_responses

    def generate_responses(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generates multiple responses (batched)."""
        B, L, _ = prompt.size()
        prompt_expanded = prompt.repeat_interleave(self.k_responses, dim=0)
        logits = self.policy_network(prompt_expanded)
        probs = F.softmax(logits, dim=-1)
        top_k = min(self.k_responses, logits.size(-1))
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        dist = Categorical(topk_probs)
        sampled_indices = dist.sample()
        response_tokens = torch.gather(topk_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
        return response_tokens.view(B, self.k_responses, L)

class PreferenceComputer:
    """Computes group preferences."""
    def __init__(self, reward_model: Callable):
        self.reward_model = reward_model

    def compute_group_preference(self, response: torch.Tensor, response_group: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """Computes batched group preference P(y ≻ G_x\{y}|x)."""
        B, K, L = response_group.shape
        response_expanded = response.unsqueeze(1)
        response_group_expanded = response_group.unsqueeze(0)
        prompt_expanded = prompt.unsqueeze(1)
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
        for b in range(B):
            for k in range(K):
                if torch.equal(response[b], response_group[b, k]):
                    mask[b, k] = 0
                    break
        response_pref = (preferences * mask).sum(dim=1) / (self.k_responses - 1 + 1e-8)
        avg_group_pref = preferences.mean(dim=1)
        advantage = response_pref - avg_group_pref - baseline
        return advantage

class SPGPOEnvironment:
    """Environment for SPGPO training, orchestrating response generation, preference, and advantage computation."""
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
        """Performs a batched step in the SPGPO environment using modular components."""
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

        ratios = torch.exp(all_current_log_probs - old_log_probs_batch)
        surr1 = ratios * all_advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * all_advantages
        loss = -torch.min(surr1, surr2).mean()
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

    env = SPGPOEnvironment(agent.policy_network, prompt_distribution, reward_model, k_responses, tokenizer, clip_ratio=0.2) # Pass policy_network

    print("Starting SPGPO Training...")
    for episode in range(num_episodes):
        episode_loss = spgpo_training_episode(agent, env, optimizer, scheduler, writer, episode, batch_size)
        print(f"Episode {episode+1}/{num_episodes}, Loss: {episode_loss:.4f}")

    writer.close()
    print("SPGPO Training Finished!")
    return agent

##############################################
# Main Function (Example Usage)
##############################################

def main():
    # Configuration for FastAttention
    config = FastAttentionConfig(
        d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
        rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
        lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
        hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4, dropout=0.1, intermediate_dim=2048, use_rff=True)

    # --- Mockup functions and data ---
    def mockup_prompt_distribution(batch_size):
        return torch.randn(batch_size, 128, config.d_model)

    def mockup_reward_model(x, y):
        if len(y.shape) == 2:
            return torch.tensor([y_item.size(0) * 0.1 for y_item in y], dtype=torch.float32)
        else:
            return torch.tensor(y.size(0) * 0.1)

    def mockup_tokenizer(text):
        tokens = text.split()
        return torch.tensor([hash(token) % 1000 for token in tokens])

    # --- Train SPGPO Agent ---
    trained_agent = train_spgpo(config, mockup_prompt_distribution, mockup_reward_model,
                                 mockup_tokenizer, num_episodes=5, batch_size=16, k_responses=4)

    # --- Example Inference ---
    prompt = torch.randn(1, 128, config.d_model)
    response = trained_agent(prompt)
    print("\nGenerated Response Shape:", response.shape)

if __name__ == "__main__":
    main()
