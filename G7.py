#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統合スクリプト
  - 設定: FastAttentionConfig, TrainingConfig
  - ユーティリティ: optimized_sqrt, create_logger
  - Attention系クラス: LowRankLinear, RandomFourierFeatures, LSHTable, CandidateFinder, FastAttention, FeedForwardNetwork, FastAttentionEncoderLayer
  - モデル: PolicyNetworkWithAttention, realistic_reward_model
  - エージェント: RolloutBuffer, SPPO
  - トレーニングループ（main）
  - APIサーバー (FastAPI)
  
実行例:
  $ python script.py         # トレーニングモード
  $ python script.py api     # APIサーバーモード (uvicornが必要)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import time

# ===============================
# 設定クラス (config)
# ===============================
@dataclass
class FastAttentionConfig:
    d_model: int          # モデル全体の次元数
    d_key: int            # Keyの次元数
    d_query: int          # Queryの次元数
    n_heads: int          # マルチヘッド注意機構のヘッド数
    rank: int             # 低ランク近似のランク
    rff_dim: int          # ランダムフーリエ特徴の次元数
    k_max: int            # クエリごとの最大候補キー数
    stride: int           # （未使用）Trie検索用のストライド長
    lsh_buckets: int      # LSHのバケット数
    lsh_bandwidth: float  # LSHのバンド幅パラメータ
    lsh_key_dim: int      # LSHに入力する特徴次元
    wu_manber_prefix_len: int  # （未使用）Wu-Manber法のプレフィックス長
    dropout: float = 0.1         # ドロップアウト率
    intermediate_dim: int = 2048 # FeedForwardの中間層次元
    use_rff: bool = True         # ランダムフーリエ特徴を使うか
    hyper_cuts_dim_groups: Optional[List[int]] = None  # HyperCuts用の次元グループ
    n_lsh_hashes: int = 4        # LSHのハッシュ関数数

@dataclass
class TrainingConfig:
    state_dim: int        # 状態空間の次元数
    action_dim: int       # 行動空間の次元数
    hidden_dim: int = 256       # ポリシーネットワークの隠れ層次元
    policy_lr: float = 3e-4     # ポリシーネットワークの学習率
    eta: float = 0.01            # ステップサイズ調整パラメータ
    alpha: float = 0.1           # 信頼領域サイズ調整パラメータ
    num_samples: int = 64        # 一度にサンプリングするデータ数（バッチサイズ）
    gamma: float = 0.99          # 割引率
    gae_lambda: float = 0.95     # GAE計算用のλ
    clip_eps: float = 0.2        # PPOクリッピング範囲
    entropy_coef: float = 0.01   # エントロピーボーナス係数
    value_coef: float = 0.5      # 価値関数の損失係数
    max_grad_norm: float = 0.5   # 勾配クリッピングの上限
    use_attention: bool = True   # 注意機構を使用するか
    seq_len: int = 8             # 注意機構に入力するシーケンス長
    use_reward_normalization: bool = True  # 報酬正規化
    n_updates: int = 10          # 1バッチあたりの更新回数
    total_timesteps: int = 1000000   # 総タイムステップ数
    eval_frequency: int = 10000      # 評価頻度
    save_frequency: int = 50000      # モデル保存頻度
    log_frequency: int = 1000        # ログ出力頻度
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# ユーティリティ関数 (utils)
# ===============================
def optimized_sqrt(n: int) -> float:
    """nが2の冪乗なら高速に平方根を計算する。"""
    if n & (n - 1) == 0:
        k = n.bit_length() - 1
        return 2 ** (k / 2)
    return math.sqrt(n)

def create_logger(log_dir: str = "./logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("RL_LOGGER")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ===============================
# Attention系クラス (attention)
# ===============================
class LowRankLinear(nn.Module):
    """低ランク近似による線形変換."""
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.U = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, out_features) * 0.02)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.U @ self.V

class RandomFourierFeatures(nn.Module):
    """ランダムフーリエ特徴によるカーネル近似."""
    def __init__(self, input_dim: int, rff_dim: int):
        super().__init__()
        self.register_buffer("omega", torch.randn(input_dim, rff_dim))
        self.register_buffer("bias", torch.rand(rff_dim) * 2 * math.pi)
        self.scale = math.sqrt(2.0 / rff_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.omega + self.bias
        return torch.cos(proj) * self.scale

class LSHTable(nn.Module):
    """LSHテーブル."""
    def __init__(self, dim: int, n_buckets: int, bandwidth: float, n_hashes: int):
        super().__init__()
        self.dim = dim
        self.n_buckets = n_buckets
        self.bandwidth = bandwidth
        self.n_hashes = n_hashes
        self.register_buffer("random_vectors", torch.randn(dim, n_hashes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.random_vectors
        return torch.floor(proj / self.bandwidth) % self.n_buckets

class CandidateFinder(nn.Module):
    """LSHを用いて候補キーのインデックスを取得する."""
    def __init__(self, config, lsh_tables: nn.ModuleList):
        super().__init__()
        self.config = config
        self.lsh_tables = lsh_tables
        # 事前に計算したマスクをキャッシュ
        self.register_buffer("mask_cache", None)
        
    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, head_idx: int) -> torch.Tensor:
        B, L, _ = query.size()
        device = query.device
        
        # バイナリ量子化の高速化
        query_bin = self.binary_quantize(query)
        key_bin = self.binary_quantize(key)
        
        # バッチ処理の効率化
        query_hash = self.lsh_tables[head_idx][0](query_bin.view(B * L, -1)).view(B, L, -1)
        key_hash = self.lsh_tables[head_idx][0](key_bin.view(B * L, -1)).view(B, L, -1)
        
        # ベクトル化されたハッシュマッチング
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=device)
        
        # より効率的なハッシュマッチング実装
        for b in range(B):
            # 全クエリハッシュを一度に拡張
            q_hash_expanded = query_hash[b].unsqueeze(1)  # [L, 1, hash_dim]
            k_hash_expanded = key_hash[b].unsqueeze(0)    # [1, L, hash_dim]
            
            # 一度に全マッチを計算
            matches_matrix = (q_hash_expanded == k_hash_expanded).all(dim=2)  # [L_q, L_k]
            
            for i in range(L):
                matches = matches_matrix[i].nonzero(as_tuple=False).squeeze(1)
                if matches.numel() > 0:
                    num_matches = min(matches.size(0), self.config.k_max)
                    candidates[b, i, :num_matches] = matches[:num_matches]
                    
        return candidates

class FastAttention(nn.Module):
    """
    Efficient attention mechanism using LSH and low-rank transformations.
    
    This implementation provides an approximation of full attention by:
    1. Using locality-sensitive hashing (LSH) to find relevant key-query pairs
    2. Employing low-rank approximations for value projections
    3. Optionally using random Fourier features for kernel approximation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Project inputs to query, key, and value spaces
        self.query_proj = nn.Linear(config.d_model, config.d_query * config.n_heads)
        self.key_proj = nn.Linear(config.d_model, config.d_key * config.n_heads)
        self.value_proj = nn.Linear(config.d_model, config.d_key * config.n_heads)
        
        # Low-rank approximations for value projections
        self.value_up_projs = nn.ModuleList([
            LowRankLinear(config.d_key, config.d_model, config.rank) 
            for _ in range(config.n_heads)
        ])
        
        # Optional random Fourier feature encoders for kernel approximation
        if config.use_rff:
            self.rff_encoders = nn.ModuleList([
                RandomFourierFeatures(config.d_key, config.rff_dim) 
                for _ in range(config.n_heads)
            ])
        else:
            self.rff_encoders = None
        
        # LSH tables for efficient key-query matching
        self.lsh_tables = nn.ModuleList([
            nn.ModuleList([
                LSHTable(
                    config.lsh_key_dim, 
                    config.lsh_buckets, 
                    config.lsh_bandwidth, 
                    config.n_lsh_hashes
                )
            ]) for _ in range(config.n_heads)
        ])
        
        # Candidate finder using LSH
        self.candidate_finder = CandidateFinder(config, self.lsh_tables)
        
        # Output projection and dropout
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize parameters with better defaults
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize weights with appropriate scaling for better gradient flow."""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, query, key, value, mask=None):
        """
        Apply fast attention mechanism.
        
        Args:
            query: Input query tensor [B, L, D]
            key: Input key tensor [B, L, D]
            value: Input value tensor [B, L, D]
            mask: Optional attention mask [B, L, L]
            
        Returns:
            Attention output tensor [B, L, D]
        """
        B, L, _ = query.size()
        
        # Project inputs to multi-head query, key, value spaces
        q = self.query_proj(query).view(B, L, self.config.n_heads, self.config.d_query)
        k = self.key_proj(key).view(B, L, self.config.n_heads, self.config.d_key)
        v = self.value_proj(value).view(B, L, self.config.n_heads, self.config.d_key)
        
        # Rearrange for multi-head attention [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        head_outputs = []
        for h in range(self.config.n_heads):
            # Find candidate keys for each query using LSH
            candidates = self.candidate_finder(q[:, h], k[:, h], h)
            cand_mask = (candidates != -1)
            
            # Replace invalid indices with 0 for safe indexing
            safe_candidates = candidates.clone()
            safe_candidates[~cand_mask] = 0
            
            # Generate batch indices for gathering
            b_idx = torch.arange(B, device=q.device).view(B, 1, 1).expand_as(safe_candidates)
            
            # Gather candidate keys and values
            candidate_keys = k[:, h][b_idx, safe_candidates]
            candidate_values = v[:, h][b_idx, safe_candidates]
            
            # Apply random Fourier features if enabled
            if self.config.use_rff and self.rff_encoders is not None:
                q_h = self.rff_encoders[h](q[:, h].reshape(-1, self.config.d_key))
                q_h = q_h.view(B, L, 1, self.config.rff_dim)
                
                candidate_keys = self.rff_encoders[h](candidate_keys.reshape(-1, self.config.d_key))
                candidate_keys = candidate_keys.view(B, L, self.config.k_max, self.config.rff_dim)
                
                scale = optimized_sqrt(self.config.rff_dim)
            else:
                q_h = q[:, h].unsqueeze(2)
                scale = optimized_sqrt(self.config.d_key)
            
            # Compute attention scores
            sim = torch.matmul(q_h, candidate_keys.transpose(-2, -1)).squeeze(2) / scale
            
            # Apply candidate mask
            sim = sim.masked_fill(~cand_mask, float('-inf'))
            
            # Apply attention mask if provided
            if mask is not None:
                # This is a simplification - actual mask application would need adaptation
                # for the sparse attention pattern used here
                pass
            
            # Compute softmax and apply dropout
            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Project values and apply attention weights
            candidate_values_proj = self.value_up_projs[h](candidate_values.view(-1, self.config.d_key))
            candidate_values_proj = candidate_values_proj.view(B, L, self.config.k_max, self.config.d_model)
            
            # Sum weighted values
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values_proj, dim=2)
            head_outputs.append(head_out)
        
        # Concatenate head outputs and project to output dimension
        concat = torch.cat(head_outputs, dim=-1)
        output = self.output_proj(concat)
        
        return self.dropout(output)


class FeedForwardNetwork(nn.Module):
    """2層のフィードフォワードネットワーク."""
    def __init__(self, config):
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
    """FastAttentionとFFNを組み合わせたエンコーダレイヤー."""
    def __init__(self, config):
        super().__init__()
        self.self_attn = FastAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        residual = src
        src = self.norm1(src)
        attn_out = self.self_attn(src, src, src)
        src = residual + self.dropout(attn_out)
        residual = src
        src = self.norm2(src)
        ffn_out = self.feed_forward(src)
        return residual + self.dropout(ffn_out)

# ===============================
# モデル (models)
# ===============================
class PolicyNetworkWithAttention(nn.Module):
    """
    注意機構を組み込んだポリシーネットワーク。
    状態エンコーディングと、方策および価値の出力ヘッドを持つ。
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 attention_config: Optional[FastAttentionConfig] = None, seq_len: int = 1):
        super().__init__()
        self.seq_len = seq_len
        if attention_config is not None:
            self.attention = FastAttentionEncoderLayer(attention_config)
            self.effective_state_dim = attention_config.d_model
            self.state_encoder = nn.Linear(state_dim, attention_config.d_model) if state_dim != attention_config.d_model else None
        else:
            self.attention = None
            self.effective_state_dim = state_dim
            self.state_encoder = None
        self.policy_head = nn.Sequential(
            nn.Linear(self.effective_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.effective_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.attention is not None:
            if state.dim() == 2:
                batch_size = state.size(0)
                state = state.view(batch_size // self.seq_len, self.seq_len, -1)
            if self.state_encoder is not None:
                state = self.state_encoder(state)
            state_features = self.attention(state)
            state_features = state_features.mean(dim=1)
        else:
            state_features = state.mean(dim=1) if state.dim() == 3 else state
        policy_out = self.policy_head(state_features)
        mean, log_std = torch.split(policy_out, policy_out.size(-1)//2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        value = self.value_head(state_features)
        return mean, std, value.squeeze(-1)

def realistic_reward_model(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    batch_size, state_dim = states.size(0), states.size(1)
    action_dim = actions.size(1)
    device = states.device
    rewards = torch.zeros(batch_size, device=device)
    target_direction = torch.ones(state_dim, device=device) / math.sqrt(state_dim)
    state_projection = (states * target_direction).sum(dim=1)
    action_alignment = torch.zeros(batch_size, device=device)
    if action_dim >= 2:
        desired_direction = F.normalize(states[:, :min(state_dim, action_dim)], dim=1)
        normalized_actions = F.normalize(actions, dim=1)
        action_alignment = (normalized_actions * desired_direction[:, :action_dim]).sum(dim=1)
    goal_reward = 2.0 * action_alignment + 0.5 * torch.tanh(state_projection)
    rewards += goal_reward
    energy_penalty = -0.1 * (actions.pow(2).sum(dim=1))
    rewards += energy_penalty
    exploration_bonus = 0.01 * torch.randn(batch_size, device=device)
    rewards += exploration_bonus
    return rewards

# ===============================
# エージェント (agent)
# ===============================
class RolloutBuffer:
    """軌跡を保存するバッファ."""
    def __init__(self, config: TrainingConfig):
        self.max_size = config.num_samples
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.device = config.device
        self.reset()
    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor):
        if self.ptr < self.max_size:
            self.states[self.ptr] = state.to(self.device)
            self.actions[self.ptr] = action.to(self.device)
            self.rewards[self.ptr] = reward.to(self.device)
            self.ptr += 1
    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.ptr
        states = self.states[:size]
        actions = self.actions[:size]
        rewards = self.rewards[:size]
        if size > 1:
            rewards_mean = rewards.mean()
            rewards_std = rewards.std() + 1e-8
            rewards = (rewards - rewards_mean) / rewards_std
        return states, actions, rewards
    def reset(self):
        self.states  = torch.zeros((self.max_size, self.state_dim), device=self.device)
        self.actions = torch.zeros((self.max_size, self.action_dim), device=self.device)
        self.rewards = torch.zeros((self.max_size, 1), device=self.device)
        self.ptr = 0

class SPPO:
    """
    Simplified Proximal Policy Optimization (SPPO) agent.
    
    This implementation provides:
    1. Gaussian policy with parameterized mean and standard deviation
    2. PPO-Clip objective for stable policy updates
    3. Attention-based neural network architecture (optional)
    4. GAE-λ advantage estimation
    """
    def __init__(self, config, attention_config=None, use_adam_optimizer=True):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize policy network
        self.policy = PolicyNetworkWithAttention(
            config.state_dim, 
            config.action_dim,
            hidden_dim=config.hidden_dim,
            attention_config=attention_config if config.use_attention else None,
            seq_len=config.seq_len
        ).to(self.device)
        
        # Create baseline policy (target network)
        self.baseline_policy = PolicyNetworkWithAttention(
            config.state_dim, 
            config.action_dim,
            hidden_dim=config.hidden_dim,
            attention_config=attention_config if config.use_attention else None,
            seq_len=config.seq_len
        ).to(self.device)
        self.baseline_policy.load_state_dict(self.policy.state_dict())
        self.baseline_policy.eval()
        
        # Hyperparameters for adaptive learning
        self.eta = config.eta
        self.alpha = config.alpha
        
        # Initialize optimizer
        if use_adam_optimizer:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        else:
            self.optimizer = None
            
        # Initialize rollout buffer
        self.buffer = RolloutBuffer(config)
        
        # Initialize statistics tracking
        self.stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl_divergence": [],
            "explained_variance": [],
            "learning_rate": config.policy_lr,
        }
        
    def normalize_state(self, x, eps=1e-8):
        """
        Normalize state vectors for improved training stability.
        
        Args:
            x: Input state tensor
            eps: Small epsilon to avoid division by zero
            
        Returns:
            Normalized state tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        return (x - mean) / std
        
    def sample_action(self, state, deterministic=False):
        """
        Sample action from current policy.
        
        Args:
            state: Current state tensor
            deterministic: If True, return mean of distribution instead of sample
            
        Returns:
            action: Sampled action
            log_prob: Log probability of sampled action
            info: Dictionary with additional information
        """
        state = state.to(self.device)
        with torch.no_grad():
            mean, std, value = self.policy(state)
            
            if deterministic:
                action = mean
            else:
                # Use reparameterization trick for backpropagation
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()
                
            # Compute log probability and entropy
            log_prob = self._compute_log_prob(action, mean, std)
            entropy = dist.entropy().sum(dim=-1)
            
        info = {
            "mean": mean.cpu().detach(),
            "std": std.cpu().detach(),
            "value": value.cpu().detach(),
            "entropy": entropy.cpu().detach()
        }
        
        return action.cpu().detach(), log_prob.cpu().detach(), info
    
    def update_baseline(self):
        """Update baseline (target) policy with current policy weights."""
        self.baseline_policy.load_state_dict(self.policy.state_dict())
    
    def _compute_log_prob(self, actions, mean, std):
        """
        Compute log probability of actions under Gaussian policy.
        
        Args:
            actions: Action tensor
            mean: Mean of Gaussian distribution
            std: Standard deviation of Gaussian distribution
            
        Returns:
            Log probability tensor
        """
        var = std.pow(2)
        log_prob = -0.5 * (
            (actions - mean)**2 / (var + 1e-8) + 
            2 * torch.log(std) + 
            math.log(2 * math.pi)
        )
        return log_prob.sum(dim=-1)
    
    def compute_advantages(self, rewards, values, next_values=None, dones=None):
        """
        Compute advantages using Generalized Advantage Estimation (GAE-λ).
        
        Args:
            rewards: Rewards tensor [B]
            values: Value estimates tensor [B]
            next_values: Next state value estimates [B] (optional)
            dones: Terminal state indicators [B] (optional)
            
        Returns:
            advantages: Advantage estimates [B]
        """
        # For simplified implementation, just use rewards as advantages
        # In a complete implementation, we would use GAE-λ here
        if next_values is None or dones is None:
            return rewards - values
            
        # GAE-λ calculation
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Work backwards through trajectory
        for t in reversed(range(len(rewards))):
            # For simplicity, assuming all trajectories are same length
            # In practice, would need to handle variable-length episodes
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        return advantages
    
    def compute_ppo_loss(self, states, actions, rewards, advantages=None):
        """
        Compute PPO-Clip objective.
        
        Args:
            states: State tensor [B, D]
            actions: Action tensor [B, A]
            rewards: Reward tensor [B]
            advantages: Advantage tensor [B] (optional)
            
        Returns:
            total_loss: Combined loss for optimization
        """
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        
        # Use rewards as advantages if not provided
        if advantages is None:
            advantages = rewards
        else:
            advantages = advantages.to(self.device)
        
        # Get current policy distribution parameters and value estimates
        mean, std, values = self.policy(states)
        current_log_prob = self._compute_log_prob(actions, mean, std)
        
        # Get baseline policy distribution parameters
        with torch.no_grad():
            base_mean, base_std, _ = self.baseline_policy(states)
            base_log_prob = self._compute_log_prob(actions, base_mean, base_std)
            
            # Compute KL divergence between old and new policy
            kl_div = self._compute_kl_divergence(base_mean, base_std, mean, std)
        
        # Compute importance sampling ratio
        ratios = torch.exp(current_log_prob - base_log_prob)
        
        # Clip ratios
        clip_eps = self.config.clip_eps
        clipped_ratios = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
        
        # Compute surrogate objectives
        surrogate1 = ratios * advantages
        surrogate2 = clipped_ratios * advantages
        
        # PPO-Clip policy loss (negative because we're minimizing)
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Value function loss
        value_loss = F.mse_loss(values, rewards.squeeze(-1))
        
        # Entropy bonus (negative because we're minimizing)
        dist = torch.distributions.Normal(mean, std)
        entropy_bonus = -dist.entropy().sum(dim=-1).mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.config.value_coef * value_loss + 
            self.config.entropy_coef * entropy_bonus
        )
        
        # Update statistics
        self.stats["policy_loss"].append(policy_loss.item())
        self.stats["value_loss"].append(value_loss.item())
        self.stats["entropy"].append(-entropy_bonus.item())
        self.stats["kl_divergence"].append(kl_div.mean().item())
        
        return total_loss
    
    def _compute_kl_divergence(self, mean1, std1, mean2, std2):
        """
        Compute KL divergence between two Gaussian distributions.
        
        Args:
            mean1, std1: Parameters of first distribution
            mean2, std2: Parameters of second distribution
            
        Returns:
            KL divergence tensor
        """
        var1 = std1.pow(2)
        var2 = std2.pow(2)
        
        kl_div = (
            torch.log(std2 / std1) + 
            (var1 + (mean1 - mean2).pow(2)) / (2 * var2) - 
            0.5
        ).sum(dim=-1)
        
        return kl_div
        
    def update_policy(self, loss):
        """
        Update policy parameters using computed loss.
        
        Args:
            loss: Loss tensor to differentiate
        """
        if self.optimizer is None:
            return
            
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(
            self.policy.parameters(), 
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        
    def adjust_hyperparameters(self, current_loss, previous_loss):
        """
        Adjust adaptive hyperparameters based on loss change.
        
        Args:
            current_loss: Current iteration loss
            previous_loss: Previous iteration loss
        """
        if current_loss < previous_loss:
            # Increase step size and trust region when improving
            self.eta *= 1.05
            self.alpha *= 1.05
        else:
            # Decrease step size and trust region when not improving
            self.eta *= 0.9
            self.alpha *= 0.9
    
    def get_stats(self):
        """Get training statistics as dictionary."""
        # Calculate additional metrics
        if len(self.stats["policy_loss"]) > 0:
            recent_stats = {
                key: np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
                for key, values in self.stats.items()
                if isinstance(values, list) and len(values) > 0
            }
            recent_stats.update({
                key: value 
                for key, value in self.stats.items()
                if not isinstance(value, list)
            })
            return recent_stats
        return self.stats
        
    def save_model(self, filepath):
        """
        Save model parameters to file.
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'stats': self.stats,
                'hyperparams': {
                    'eta': self.eta,
                    'alpha': self.alpha
                }
            }, filepath)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")
    
    def load_model(self, filepath):
        """
        Load model parameters from file.
        
        Args:
            filepath: Path to load model from
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            if 'stats' in checkpoint:
                self.stats = checkpoint['stats']
                
            if 'hyperparams' in checkpoint:
                self.eta = checkpoint['hyperparams'].get('eta', self.eta)
                self.alpha = checkpoint['hyperparams'].get('alpha', self.alpha)
                
            # Also update baseline policy
            self.baseline_policy.load_state_dict(self.policy.state_dict())
            print(f"Model loaded successfully from {filepath}")
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
    
    def train_mode(self):
        """Set policy network to training mode."""
        self.policy.train()
    
    def eval_mode(self):
        """Set policy network to evaluation mode."""
        self.policy.eval()


# ===============================
# トレーニングループ (main)
# ===============================
def train_agent(config: TrainingConfig, 
                attention_config: Optional[FastAttentionConfig] = None,
                custom_reward_fn = None,
                checkpoint_path: Optional[str] = None):
    """
    改善されたトレーニングループ
    
    Args:
        config: トレーニング設定
        attention_config: 注意機構の設定（オプション）
        custom_reward_fn: カスタム報酬関数（オプション）
        checkpoint_path: 続きから学習する場合のモデルパス（オプション）
    """
    # ロガーのセットアップ
    logger = create_logger("./logs")
    run_id = int(time.time())
    log_dir = os.path.join("./logs", f"run_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    
    # デバイス情報のログ記録
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
        # GPUメモリ使用量のモニタリング
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU")
    
    # 設定の保存（JSON形式）
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        config_dict = {
            "training": config.__dict__,
            "attention": attention_config.__dict__ if attention_config else None
        }
        json.dump(config_dict, f, indent=2)
    
    # エージェントの初期化
    agent = SPPO(config, attention_config)
    
    # チェックポイントから再開
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        agent.load_model(checkpoint_path)
    
    # 報酬関数の設定
    reward_fn = custom_reward_fn if custom_reward_fn else realistic_reward_model
    
    # 評価用の変数
    best_reward = float('-inf')
    previous_loss = float('inf')
    
    # トレーニング統計
    training_stats = {
        'episodes': [],
        'losses': [],
        'rewards': [],
        'eta_values': [],
        'alpha_values': []
    }
    
    logger.info("Training started")
    try:
        total_steps = 0
        episodes = config.total_timesteps // config.num_samples
        
        # トレーニングループ
        for episode in range(1, episodes + 1):
            episode_start_time = time.time()
            agent.train_mode()
            agent.buffer.reset()
            
            # サンプリングループ
            episode_rewards = []
            for step in range(config.num_samples):
                # 状態の生成（実際の環境ではここを環境とのインタラクションに置き換え）
                state = torch.randn(config.num_samples, config.state_dim)
                norm_state = agent.normalize_state(state)
                
                # 行動のサンプリング
                action, _, info = agent.sample_action(norm_state)
                
                # 報酬の計算
                reward = reward_fn(norm_state, action)
                episode_rewards.append(reward.mean().item())
                
                # バッファに追加
                agent.buffer.add(norm_state, action, reward)
                
                total_steps += 1
                
                # ログ記録
                if total_steps % config.log_frequency == 0:
                    logger.info(f"Episode {episode}, Step {step}, TotalSteps {total_steps}, "
                               f"AvgReward: {np.mean(episode_rewards):.4f}")
            
            # バッファからデータを取得し、ポリシーを更新
            states_batch, actions_batch, rewards_batch = agent.buffer.get()
            
            # 複数回の更新（エポック）
            epoch_losses = []
            for _ in range(config.n_updates):
                loss = agent.compute_ppo_loss(states_batch, actions_batch, rewards_batch)
                agent.update_policy(loss)
                epoch_losses.append(loss.item())
            
            # ベースラインモデルの更新
            agent.update_baseline()
            
            # 現在の損失（最終エポックの損失）
            current_loss = epoch_losses[-1]
            
            # ハイパーパラメータの調整
            agent.adjust_hyperparameters(current_loss, previous_loss)
            previous_loss = current_loss
            
            # エピソード統計の記録
            avg_reward = np.mean(episode_rewards)
            episode_time = time.time() - episode_start_time
            logger.info(f"Episode {episode} completed in {episode_time:.2f}s, "
                       f"Loss: {current_loss:.4f}, AvgReward: {avg_reward:.4f}, "
                       f"eta: {agent.eta:.3f}, alpha: {agent.alpha:.3f}")
            
            # 統計情報の更新
            training_stats['episodes'].append(episode)
            training_stats['losses'].append(current_loss)
            training_stats['rewards'].append(avg_reward)
            training_stats['eta_values'].append(agent.eta)
            training_stats['alpha_values'].append(agent.alpha)
            
            # 定期的な評価
            if total_steps % config.eval_frequency == 0:
                eval_reward = evaluate_agent(agent, reward_fn, config)
                logger.info(f"Evaluation at step {total_steps}: AvgReward = {eval_reward:.4f}")
                
                # 最良モデルの保存
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    best_model_path = os.path.join(log_dir, "best_model.pth")
                    agent.save_model(best_model_path)
                    logger.info(f"New best model saved with reward {best_reward:.4f}")
            
            # 定期的なモデル保存
            if total_steps % config.save_frequency == 0:
                checkpoint_dir = os.path.join(log_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_path = os.path.join(checkpoint_dir, f"checkpoint_step{total_steps}.pth")
                agent.save_model(save_path)
                logger.info(f"Checkpoint saved at {save_path}")
                
                # トレーニング統計をプロット
                plot_training_stats(training_stats, os.path.join(log_dir, "training_curves.png"))
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 最終モデルの保存
        final_model_path = os.path.join(log_dir, "final_model.pth")
        agent.save_model(final_model_path)
        logger.info(f"Training finished, final model saved at {final_model_path}")
        
        # トレーニング統計の保存
        stats_path = os.path.join(log_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f)
        
        # 最終的な学習曲線のプロット
        plot_training_stats(training_stats, os.path.join(log_dir, "final_training_curves.png"))
        
        return agent, best_reward

def evaluate_agent(agent, reward_fn, config, n_episodes=5):
    """
    エージェントの評価を行う
    
    Args:
        agent: 評価するエージェント
        reward_fn: 報酬関数
        config: 設定
        n_episodes: 評価エピソード数
    
    Returns:
        平均報酬
    """
    agent.eval_mode()
    all_rewards = []
    
    for _ in range(n_episodes):
        episode_rewards = []
        for _ in range(10):  # 各エピソードは10ステップとする
            state = torch.randn(1, config.state_dim)
            norm_state = agent.normalize_state(state)
            action, _, _ = agent.sample_action(norm_state, deterministic=True)
            reward = reward_fn(norm_state, action)
            episode_rewards.append(reward.item())
        
        all_rewards.append(np.mean(episode_rewards))
    
    agent.train_mode()
    return np.mean(all_rewards)

def plot_training_stats(stats, save_path):
    """トレーニング統計をプロットする"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # 損失曲線
        axs[0, 0].plot(stats['episodes'], stats['losses'])
        axs[0, 0].set_title('Loss Curve')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Loss')
        
        # 報酬曲線
        axs[0, 1].plot(stats['episodes'], stats['rewards'])
        axs[0, 1].set_title('Average Reward')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Reward')
        
        # ハイパーパラメータ曲線
        axs[1, 0].plot(stats['episodes'], stats['eta_values'])
        axs[1, 0].set_title('Learning Rate (eta)')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('eta')
        
        axs[1, 1].plot(stats['episodes'], stats['alpha_values'])
        axs[1, 1].set_title('Trust Region Size (alpha)')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('alpha')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except ImportError:
        print("matplotlib not installed, skipping plot generation")


# ===============================
# APIサーバー (FastAPI)
# ===============================
def run_api(host="0.0.0.0", port=8000, model_path="./models/final_model.pth"):
    """
    改善されたAPIサーバー実装
    
    Args:
        host: ホスト名
        port: ポート番号
        model_path: 読み込むモデルのパス
    """
    try:
        from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field, validator
        import uvicorn
        from typing import List, Dict, Any, Optional
        import logging
        import time
    except ImportError:
        print("必要なパッケージがインストールされていません。")
        print("以下のコマンドでインストールしてください:")
        print("pip install fastapi uvicorn pydantic")
        return

    # リクエスト/レスポンスモデルの定義
    class StateRequest(BaseModel):
        state: List[float] = Field(..., description="入力状態ベクトル")
        
        @validator('state')
        def validate_state_dim(cls, v, values):
            if len(v) != 128:  # 期待する状態次元
                raise ValueError(f"状態ベクトルは長さ128である必要があります。現在の長さ: {len(v)}")
            return v
    
    class BatchStateRequest(BaseModel):
        states: List[List[float]] = Field(..., description="バッチ処理用の状態ベクトルのリスト")
    
    class ActionResponse(BaseModel):
        action: List[float] = Field(..., description="予測された行動ベクトル")
        value: float = Field(..., description="状態価値の推定値")
        processing_time: float = Field(..., description="処理時間（ミリ秒）")
    
    class BatchActionResponse(BaseModel):
        actions: List[List[float]] = Field(..., description="予測された行動ベクトルのリスト")
        values: List[float] = Field(..., description="状態価値の推定値のリスト")
        processing_time: float = Field(..., description="処理時間（ミリ秒）")
    
    class HealthResponse(BaseModel):
        status: str = "healthy"
        model_loaded: bool
        gpu_available: bool
        version: str = "1.0.0"
    
    # FastAPIアプリの設定
    app = FastAPI(
        title="高速注意機構付きPPOエージェントAPI",
        description="状態ベクトルを入力として、最適な行動と状態価値を予測するAPI",
        version="1.0.0"
    )
    
    # CORSミドルウェアの追加
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ロガーの設定
    api_logger = logging.getLogger("api")
    api_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    api_logger.addHandler(handler)
    
    # エージェントの初期化
    training_config = TrainingConfig(state_dim=128, action_dim=2)
    attention_config = FastAttentionConfig(
        d_model=128, d_key=32, d_query=32, n_heads=4,
        rank=16, rff_dim=64, k_max=32, stride=4,
        lsh_buckets=64, lsh_bandwidth=2.0, lsh_key_dim=32,
        wu_manber_prefix_len=4, dropout=0.1, intermediate_dim=256,
        use_rff=True, hyper_cuts_dim_groups=None, n_lsh_hashes=4
    )
    
    agent = SPPO(training_config, attention_config)
    model_loaded = False
    
    # 統計情報の追跡
    request_stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_processing_time": 0,
        "avg_processing_time": 0,
    }
    
    def get_agent():
        """依存性注入用のエージェント取得関数"""
        return agent
    
    def update_stats(success: bool, processing_time: float):
        """リクエスト統計情報を更新する"""
        request_stats["total_requests"] += 1
        if success:
            request_stats["successful_requests"] += 1
        else:
            request_stats["failed_requests"] += 1
        request_stats["total_processing_time"] += processing_time
        request_stats["avg_processing_time"] = (
            request_stats["total_processing_time"] / request_stats["total_requests"]
        )
    
    @app.on_event("startup")
    async def startup_event():
        """アプリケーション起動時の処理"""
        nonlocal model_loaded
        
        api_logger.info("APIサーバーを起動しています...")
        
        if os.path.exists(model_path):
            try:
                api_logger.info(f"モデルを読み込んでいます: {model_path}")
                agent.load_model(model_path)
                agent.eval_mode()
                model_loaded = True
                api_logger.info("モデルの読み込みが完了しました")
            except Exception as e:
                api_logger.error(f"モデルの読み込み中にエラーが発生しました: {e}")
                model_loaded = False
        else:
            api_logger.warning(f"モデルファイルが見つかりません: {model_path}")
            model_loaded = False
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """ヘルスチェックエンドポイント"""
        return HealthResponse(
            model_loaded=model_loaded,
            gpu_available=torch.cuda.is_available()
        )
    
    @app.post("/predict", response_model=ActionResponse)
    async def predict_action(
        request: StateRequest,
        background_tasks: BackgroundTasks,
        agent: SPPO = Depends(get_agent)
    ):
        """単一状態から行動を予測する"""
        if not model_loaded:
            raise HTTPException(
                status_code=503,
                detail="モデルが読み込まれていません。サーバーの準備ができていません。"
            )
        
        start_time = time.time()
        success = False
        
        try:
            # 状態テンソルの準備
            state_tensor = torch.tensor(request.state, dtype=torch.float32)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # 正規化
            normalized_state = agent.normalize_state(state_tensor)
            
            # 行動の予測
            action, _, info = agent.sample_action(normalized_state, deterministic=True)
            
            processing_time = (time.time() - start_time) * 1000  # ミリ秒に変換
            
            success = True
            response = ActionResponse(
                action=action.cpu().numpy().tolist()[0],
                value=float(info["value"].cpu().numpy()[0]),
                processing_time=processing_time
            )
            
            # 非同期で統計情報を更新
            background_tasks.add_task(update_stats, success, processing_time)
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            background_tasks.add_task(update_stats, False, processing_time)
            api_logger.error(f"予測中にエラーが発生しました: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"予測の処理中にエラーが発生しました: {str(e)}"
            )
    
    @app.post("/predict_batch", response_model=BatchActionResponse)
    async def predict_batch(
        request: BatchStateRequest,
        background_tasks: BackgroundTasks,
        agent: SPPO = Depends(get_agent)
    ):
        """複数の状態をバッチ処理で予測する"""
        if not model_loaded:
            raise HTTPException(
                status_code=503,
                detail="モデルが読み込まれていません。サーバーの準備ができていません。"
            )
        
        start_time = time.time()
        success = False


# ===============================
# エントリーポイント
# ===============================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api()
    else:
        train_agent()
