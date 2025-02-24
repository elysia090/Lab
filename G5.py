import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

#############################################
# GRPOAgent: 状態からプレフィックス長調整量を出力
# Gaussian Policy に基づき、連続値を生成後丸めて整数に変換する
#############################################
class GRPOAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, fixed_std: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.fixed_std = fixed_std

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # state: [B_state, state_dim]
        x = F.relu(self.fc1(state))
        mean = self.fc2(x)  # [B_state, action_dim]
        dist = torch.distributions.Normal(mean, self.fixed_std)
        action = dist.rsample()  # [B_state, action_dim]
        log_prob = dist.log_prob(action)  # [B_state, action_dim]
        # 連続値を丸めて整数に変換、その後 float に戻して平均を計算
        action_int = torch.round(action).to(torch.long)
        return action_int.float().mean(dim=1, keepdim=True), log_prob.mean(dim=1, keepdim=True)

#############################################
# FastAttention 用設定
#############################################
@dataclass
class FastAttentionConfig:
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

#############################################
# 基本モジュール
#############################################
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
        self.random_vectors = nn.Parameter(torch.randn(dim, n_hashes), requires_grad=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, dim]
        proj = x.matmul(self.random_vectors)  # [B, L, n_hashes]
        return torch.floor(proj / self.bandwidth) % self.n_buckets

_TRIE_INDICES_KEY = '_indices'
class Trie(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.root_node: Dict = {}
        self.stride_len = stride
    def insert(self, binary_vector: torch.Tensor, index: int) -> None:
        # binary_vector: [d]
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

#############################################
# CandidateFinder (GRPO 統合版)
#############################################
class CandidateFinder(nn.Module):
    def __init__(self, config: FastAttentionConfig, tries: List[Trie], lsh_tables: nn.ModuleList):
        super().__init__()
        self.config = config
        self.tries = tries
        self.lsh_tables = lsh_tables
        # 内部で利用する Wu-Manber のプレフィックス長
        self.current_prefix_len = config.wu_manber_prefix_len
        # GRPOエージェント: 状態次元は 10、調整対象はプレフィックス長 delta（1次元）
        self.grpo_agent = GRPOAgent(state_dim=10, action_dim=1)

    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()

    def split_features_by_dim_groups(self, features: torch.Tensor) -> List[torch.Tensor]:
        if self.config.hyper_cuts_dim_groups is None:
            return [features]
        groups = []
        start = 0
        for group_dim in self.config.hyper_cuts_dim_groups:
            groups.append(features[:, :, start:start+group_dim])
            start += group_dim
        return groups

    def _build_wu_manber_hash_table(self, key_bin: torch.Tensor) -> Dict[tuple, List[int]]:
        # key_bin: [L, d]
        table: Dict[tuple, List[int]] = {}
        L = key_bin.size(0)
        for i in range(L):
            prefix = tuple(key_bin[i, :int(self.current_prefix_len)].tolist())
            table.setdefault(prefix, []).append(i)
        return table

    def _wu_manber_search(self, query_bin: torch.Tensor, table: Dict[tuple, List[int]]) -> List[int]:
        prefix = tuple(query_bin[:int(self.current_prefix_len)].tolist())
        return table.get(prefix, [])

    def _get_wu_manber_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor) -> List[List[List[int]]]:
        # query_grp, key_grp: [B, L, d]
        B, L, d = key_grp.size()
        assert d >= self.current_prefix_len, "入力次元がプレフィックス長以上でなければなりません。"
        key_bin = self.binary_quantize(key_grp)
        query_bin = self.binary_quantize(query_grp)
        cand_lists = []
        for b in range(B):
            table = self._build_wu_manber_hash_table(key_bin[b])
            batch_list = []
            for i in range(L):
                batch_list.append(self._wu_manber_search(query_bin[b, i], table))
            cand_lists.append(batch_list)
        return cand_lists

    def _get_trie_candidates_group(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> List[List[List[int]]]:
        # query_grp, key_grp: [B, L, d]
        B, L, d = key_grp.size()
        cand_lists = []
        for b in range(B):
            trie = Trie(self.config.stride)
            key_bin = self.binary_quantize(key_grp[b])
            for i in range(L):
                trie.insert(key_bin[i], i)
            batch_list = []
            query_bin = self.binary_quantize(query_grp[b])
            for i in range(L):
                batch_list.append(trie.search(query_bin[i]))
            cand_lists.append(batch_list)
        return cand_lists

    def merge_candidate_indices_groups(self, cand_tensors: List[torch.Tensor]) -> torch.Tensor:
        merged = torch.cat(cand_tensors, dim=-1).float() # cast to float here
        merged = merged.masked_fill(merged == -1, float('inf'))
        merged, _ = torch.sort(merged, dim=-1)
        merged = torch.unique_consecutive(merged, dim=-1)
        merged = merged.masked_fill(merged == float('inf'), -1)
        B, L, N = merged.size()
        if N < self.config.k_max:
            padding = torch.full((B, L, self.config.k_max - N), -1, device=merged.device)
            merged = torch.cat([merged, padding], dim=-1)
        else:
            merged = merged[:, :, :self.config.k_max]
        return merged

    def update_hyperparams(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # state: [B_state, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action, log_prob = self.grpo_agent(state)  # [B_state, 1]
        # バッチごとの delta を保持し、平均値で更新（最低値は1）
        new_prefix_tensor = torch.clamp(self.current_prefix_len + action, min=1)
        self.current_prefix_len = new_prefix_tensor.mean().item()
        print(f"GRPO update: delta={action.mean().item():.4f}, updated prefix length={self.current_prefix_len}")
        return new_prefix_tensor, log_prob.mean()

    def _process_dimension_group_candidates(self, query_grp: torch.Tensor, key_grp: torch.Tensor, head_idx: int) -> torch.Tensor:
        # query_grp, key_grp: [B, L, d]
        B, L, d = query_grp.size()
        wu_cands = self._get_wu_manber_candidates_group(query_grp, key_grp)
        trie_cands = self._get_trie_candidates_group(query_grp, key_grp, head_idx)
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=query_grp.device)
        for b in range(B):
            for i in range(L):
                common = list(set(wu_cands[b][i]) & set(trie_cands[b][i]))
                if common:
                    common_tensor = torch.tensor(common, dtype=torch.long, device=query_grp.device)
                    count = min(common_tensor.numel(), self.config.k_max)
                    candidates[b, i, :count] = common_tensor[:count]
        return candidates

    def forward(self, query_up: torch.Tensor, key_up: torch.Tensor, head_idx: int, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # query_up, key_up: [B, L, d] (通常 d = d_key)
        if state is not None:
            new_param, log_prob = self.update_hyperparams(state)
            grpo_info = (new_param, log_prob)
        else:
            grpo_info = None
        candidates = self._process_dimension_group_candidates(query_up, key_up, head_idx)  # [B, L, k_max]
        merged = self.merge_candidate_indices_groups([candidates])
        return merged[:, :, :self.config.k_max], grpo_info

#############################################
# Oracle 候補探索: 全結合 Attention による Top-k (理想候補)
#############################################
def oracle_candidates(queries: torch.Tensor, keys: torch.Tensor, k_max: int) -> torch.Tensor:
    # queries, keys: [B, L, d]
    B, L, d = queries.shape
    queries_norm = F.normalize(queries, p=2, dim=-1)
    keys_norm = F.normalize(keys, p=2, dim=-1)
    sim = torch.matmul(queries_norm, keys_norm.transpose(-2, -1))  # [B, L, L]
    topk = torch.topk(sim, k=k_max, dim=-1)[1]  # [B, L, k_max]
    return topk

#############################################
# 報酬計算: Jaccard 類似度 (候補探索結果 vs Oracle)
#############################################
def compute_reward(candidate_out: torch.Tensor, oracle_out: torch.Tensor) -> torch.Tensor:
    # candidate_out, oracle_out: [B, L, k_max]
    B, L, k = candidate_out.shape
    rewards = []
    for b in range(B):
        for i in range(L):
            cand_set = set(candidate_out[b, i].tolist())
            oracle_set = set(oracle_out[b, i].tolist())
            cand_set.discard(-1)
            oracle_set.discard(-1)
            if not oracle_set:
                sim_val = 0.0
            else:
                intersection = len(cand_set & oracle_set)
                union = len(cand_set | oracle_set)
                sim_val = intersection / union if union > 0 else 0.0
            rewards.append(sim_val)
    return torch.tensor(rewards).mean()

#############################################
# AbsorptionProjection: クエリをキー空間に写像
#############################################
class AbsorptionProjection(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, rank: int):
        super().__init__()
        self.u_q = nn.Parameter(torch.randn(query_dim, rank) / math.sqrt(rank))
        self.v_q = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))
        self.u_k = nn.Parameter(torch.randn(key_dim, rank) / math.sqrt(rank))
        self.v_k = nn.Parameter(torch.randn(rank, key_dim) / math.sqrt(rank))
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query: [B, L, d_query], key: [B, L, d_key]
        W_UQ = torch.matmul(self.u_q, self.v_q)   # [d_query, d_key]
        W_UK = torch.matmul(self.u_k, self.v_k)     # [d_key, d_key]
        W_absorb = torch.matmul(W_UK.transpose(0,1), W_UQ)  # [d_key, d_key]
        Q_proj = torch.matmul(query, W_absorb.transpose(0,1)) # [B, L, d_key]
        return Q_proj, key

#############################################
# FastAttention 本体
#############################################
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
        self.lsh_tables_list = nn.ModuleList([
            nn.ModuleList([
                LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
                for dim in config.hyper_cuts_dim_groups
            ])
            for _ in range(config.n_heads)
        ]) if config.hyper_cuts_dim_groups else nn.ModuleList([
            nn.ModuleList([
                LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
            ]) for _ in range(config.n_heads)
        ])
        self.tries_list = nn.ModuleList([Trie(config.stride) for _ in range(config.n_heads)])
        self.candidate_finder = CandidateFinder(config, self.tries_list, self.lsh_tables_list)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query, key, value: [B, L, d_model]
        B, L, _ = query.size()
        query_down = self.query_down_proj(query)  # [B, L, d_query]
        key_down = self.key_value_down_proj(key)    # [B, L, d_key]
        head_outputs = []
        for head_idx in range(self.config.n_heads):
            Q_proj, K_proj = self.absorption_projs[head_idx](query_down, key_down)  # [B, L, d_key]
            candidates, _ = self.candidate_finder(Q_proj, K_proj, head_idx)
            cand_mask = candidates != -1  # [B, L, k_max]
            safe_candidates = candidates.clone()
            # -1 を有効範囲外のインデックス L に置換
            safe_candidates[safe_candidates == -1] = L
            b_idx = torch.arange(B, device=K_proj.device).view(B, 1, 1).expand(B, L, self.config.k_max)
            candidate_keys = K_proj[b_idx, safe_candidates]  # [B, L, k_max, d_key]
            q_exp = Q_proj.unsqueeze(2)  # [B, L, 1, d_key]
            if self.config.use_rff:
                q_exp = self.rff_encoders[head_idx](q_exp.view(-1, self.config.d_key)).view(B, L, 1, self.config.rff_dim)
                candidate_keys = self.rff_encoders[head_idx](candidate_keys.view(-1, self.config.d_key)).view(B, L, self.config.k_max, self.config.rff_dim)
                scale = math.sqrt(self.config.rff_dim)
            else:
                scale = math.sqrt(self.config.d_key)
            sim = torch.matmul(q_exp, candidate_keys.transpose(-2, -1)).squeeze(2) / scale  # [B, L, k_max]
            sim = sim.masked_fill(~cand_mask, float('-inf'))
            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)
            b_idx = torch.arange(B, device=key_down.device).view(B, 1, 1).expand(B, L, self.config.k_max)
            candidate_values = key_down[b_idx, safe_candidates]  # [B, L, k_max, d_key]
            candidate_values = self.value_up_projs[head_idx](candidate_values.view(-1, self.config.d_key)).view(B, L, self.config.k_max, self.config.d_model)
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values, dim=2)  # [B, L, d_model]
            head_outputs.append(head_out)
        concat = torch.cat(head_outputs, dim=-1)  # [B, L, d_model * n_heads]
        output = self.output_proj(concat)         # [B, L, d_model]
        return self.dropout(output)

#############################################
# FeedForwardNetwork と EncoderLayer
#############################################
class FeedForwardNetwork(nn.Module):
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

#############################################
# 学習ループ (GRPO + 候補探索統合システム)
#############################################
def compute_state(candidate_out: torch.Tensor) -> torch.Tensor:
    """
    候補探索結果 candidate_out: [B, L, k_max]
    各クエリにおける有効候補（-1 でない）の割合などを特徴量として state を構築する。
    ここでは簡単な例として、各バッチの平均有効率を 10 次元ベクトルとして返す。
    """
    B, L, k = candidate_out.shape
    valid_mask = (candidate_out != -1).float()  # [B, L, k_max]
    valid_ratio = valid_mask.mean(dim=-1)        # [B, L]
    avg_valid_ratio = valid_ratio.mean(dim=1, keepdim=True)  # [B, 1]
    # state 次元を 10 に拡張
    state = avg_valid_ratio.repeat(1, 10)
    return state

def train_candidate_finder_grpo():
    config = FastAttentionConfig(
        d_model=512, d_key=64, d_query=64, n_heads=8, rank=32,
        rff_dim=128, k_max=64, stride=4, lsh_buckets=32,
        lsh_bandwidth=4.0, lsh_key_dim=64, wu_manber_prefix_len=3,
        hyper_cuts_dim_groups=[32, 32], n_lsh_hashes=4, use_rff=True)
    # 各ヘッド用に Trie を1個ずつ用意
    tries = [Trie(config.stride) for _ in range(config.n_heads)]
    lsh_tables = nn.ModuleList([
        nn.ModuleList([
            LSHTable(dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)
            for dim in config.hyper_cuts_dim_groups
        ])
        for _ in range(config.n_heads)
    ])
    candidate_finder = CandidateFinder(config, tries, lsh_tables)
    optimizer = optim.Adam(candidate_finder.grpo_agent.parameters(), lr=0.001)
    baseline_net = nn.Linear(10, 1)
    baseline_optimizer = optim.Adam(baseline_net.parameters(), lr=0.001)

    B, L, d = 4, 128, 64  # queries, keys: [B, L, d]
    queries = torch.randn(B, L, d)
    keys = torch.randn(B, L, d)

    num_epochs = 200
    state = torch.randn(B, 10)  # 初期状態
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        baseline_optimizer.zero_grad()
        # GRPO 状態入力として前回の候補探索結果から計算した state を利用
        candidate_out, grpo_info = candidate_finder(queries, keys, head_idx=0, state=state)
        # 候補探索結果に基づいて新たな状態を計算
        state = compute_state(candidate_out)
        oracle_out = oracle_candidates(queries, keys, config.k_max)
        reward = compute_reward(candidate_out, oracle_out)
        baseline_value = baseline_net(state).mean()
        advantage = reward - baseline_value.detach()
        if grpo_info is not None:
            _, log_prob = grpo_info
            policy_loss = -log_prob * advantage
            value_loss = F.mse_loss(baseline_value, reward)
            loss = policy_loss + value_loss
        else:
            loss = torch.tensor(0.0)
        loss.backward()
        optimizer.step()
        baseline_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Reward = {reward.item():.4f}, Baseline = {baseline_value.item():.4f}")
            print(f"Current Wu-Manber Prefix Length: {candidate_finder.current_prefix_len}")

if __name__ == "__main__":
    train_candidate_finder_grpo()
