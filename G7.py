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
    def binary_quantize(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).float()
    def forward(self, query: torch.Tensor, key: torch.Tensor, head_idx: int) -> torch.Tensor:
        B, L, _ = query.size()
        device = query.device
        query_bin = self.binary_quantize(query)
        key_bin   = self.binary_quantize(key)
        query_hash = self.lsh_tables[head_idx][0](query_bin.view(B * L, -1)).view(B, L, -1)
        key_hash   = self.lsh_tables[head_idx][0](key_bin.view(B * L, -1)).view(B, L, -1)
        candidates = torch.full((B, L, self.config.k_max), -1, dtype=torch.long, device=device)
        for b in range(B):
            for i in range(L):
                matches = (query_hash[b, i].unsqueeze(0) == key_hash[b]).all(dim=1).nonzero(as_tuple=False).squeeze(1)
                if matches.numel() > 0:
                    num_matches = min(matches.size(0), self.config.k_max)
                    candidates[b, i, :num_matches] = matches[:num_matches]
        return candidates

class FastAttention(nn.Module):
    """LSHと低ランク変換を用いた高速注意機構."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query_proj = nn.Linear(config.d_model, config.d_query * config.n_heads)
        self.key_proj   = nn.Linear(config.d_model, config.d_key * config.n_heads)
        self.value_proj = nn.Linear(config.d_model, config.d_key * config.n_heads)
        self.value_up_projs = nn.ModuleList([
            LowRankLinear(config.d_key, config.d_model, config.rank) for _ in range(config.n_heads)
        ])
        self.rff_encoders = nn.ModuleList([
            RandomFourierFeatures(config.d_key, config.rff_dim) for _ in range(config.n_heads)
        ]) if config.use_rff else None
        self.lsh_tables = nn.ModuleList([
            nn.ModuleList([LSHTable(config.lsh_key_dim, config.lsh_buckets, config.lsh_bandwidth, config.n_lsh_hashes)])
            for _ in range(config.n_heads)
        ])
        self.candidate_finder = CandidateFinder(config, self.lsh_tables)
        self.output_proj = nn.Linear(config.d_model * config.n_heads, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = query.size()
        q = self.query_proj(query).view(B, L, self.config.n_heads, self.config.d_query)
        k = self.key_proj(key).view(B, L, self.config.n_heads, self.config.d_key)
        v = self.value_proj(value).view(B, L, self.config.n_heads, self.config.d_key)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        head_outputs = []
        for h in range(self.config.n_heads):
            candidates = self.candidate_finder(q[:, h], k[:, h], h)
            cand_mask = (candidates != -1)
            safe_candidates = candidates.clone()
            safe_candidates[safe_candidates == -1] = 0
            b_idx = torch.arange(B, device=q.device).view(B, 1, 1).expand_as(safe_candidates)
            candidate_keys = k[:, h][b_idx, safe_candidates]
            candidate_values = v[:, h][b_idx, safe_candidates]
            if self.config.use_rff and self.rff_encoders is not None:
                q_h = self.rff_encoders[h](q[:, h].reshape(-1, self.config.d_key)).view(B, L, 1, self.config.rff_dim)
                candidate_keys = self.rff_encoders[h](candidate_keys.reshape(-1, self.config.d_key)).view(B, L, self.config.k_max, self.config.rff_dim)
                scale = optimized_sqrt(self.config.rff_dim)
            else:
                q_h = q[:, h].unsqueeze(2)
                scale = optimized_sqrt(self.config.d_key)
            sim = torch.matmul(q_h, candidate_keys.transpose(-2, -1)).squeeze(2) / scale
            sim = sim.masked_fill(~cand_mask, float('-inf'))
            attn_weights = F.softmax(sim, dim=-1)
            attn_weights = self.dropout(attn_weights)
            candidate_values_proj = self.value_up_projs[h](candidate_values.view(-1, self.config.d_key))
            candidate_values_proj = candidate_values_proj.view(B, L, self.config.k_max, self.config.d_model)
            head_out = torch.sum(attn_weights.unsqueeze(-1) * candidate_values_proj, dim=2)
            head_outputs.append(head_out)
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
    """簡易PPOエージェント."""
    def __init__(self, config: TrainingConfig, attention_config: Optional[FastAttentionConfig] = None,
                 use_adam_optimizer: bool = True):
        self.device = torch.device(config.device)
        self.policy = PolicyNetworkWithAttention(config.state_dim, config.action_dim,
                                                 hidden_dim=config.hidden_dim,
                                                 attention_config=attention_config if config.use_attention else None,
                                                 seq_len=config.seq_len).to(self.device)
        self.baseline_policy = PolicyNetworkWithAttention(config.state_dim, config.action_dim,
                                                          hidden_dim=config.hidden_dim,
                                                          attention_config=attention_config if config.use_attention else None,
                                                          seq_len=config.seq_len).to(self.device)
        self.baseline_policy.load_state_dict(self.policy.state_dict())
        self.baseline_policy.eval()
        self.config = config
        self.eta = config.eta
        self.alpha = config.alpha
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.policy_lr) if use_adam_optimizer else None
        self.buffer = RolloutBuffer(config)
    def normalize_state(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        return (x - mean) / std
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        state = state.to(self.device)
        with torch.no_grad():
            mean, std, value = self.policy(state)
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        info = {"mean": mean, "std": std, "value": value, "entropy": entropy}
        return action.detach(), log_prob.detach(), info
    def update_baseline(self):
        self.baseline_policy.load_state_dict(self.policy.state_dict())
    def compute_log_prob(self, actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        var = std.pow(2)
        log_prob = -0.5 * ((actions - mean)**2 / (var + 1e-8) + 2 * std.log() + math.log(2 * math.pi))
        return log_prob.sum(dim=-1)
    def compute_ppo_loss(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                         advantages: Optional[torch.Tensor] = None) -> torch.Tensor:
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        if advantages is None:
            advantages = rewards
        mean, std, values = self.policy(states)
        log_prob = self.compute_log_prob(actions, mean, std)
        with torch.no_grad():
            base_mean, base_std, _ = self.baseline_policy(states)
            base_log_prob = self.compute_log_prob(actions, base_mean, base_std)
        ratios = torch.exp(log_prob - base_log_prob)
        clipped_ratios = torch.clamp(ratios, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
        value_loss = ((values - rewards.squeeze(-1))**2).mean()
        entropy_bonus = - (log_prob * torch.exp(log_prob)).mean()
        total_loss = policy_loss + self.config.value_coef * value_loss + self.config.entropy_coef * entropy_bonus
        return total_loss
    def update_policy(self, loss: torch.Tensor):
        if self.optimizer is None:
            return
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
    def adjust_hyperparameters(self, current_loss: float, previous_loss: float):
        if current_loss < previous_loss:
            self.eta *= 1.05
            self.alpha *= 1.05
        else:
            self.eta *= 0.9
            self.alpha *= 0.9
    def save_model(self, filepath: str):
        try:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
            }, filepath)
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")
    def load_model(self, filepath: str):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.baseline_policy.load_state_dict(self.policy.state_dict())
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
    def train_mode(self):
        self.policy.train()
    def eval_mode(self):
        self.policy.eval()

# ===============================
# トレーニングループ (main)
# ===============================
def train_agent():
    training_config = TrainingConfig(
        state_dim=128,
        action_dim=2,
        hidden_dim=256,
        num_samples=8,
        total_timesteps=5000,
        eval_frequency=1000,
        save_frequency=2000,
        log_frequency=100,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    attention_config = FastAttentionConfig(
        d_model=128, d_key=32, d_query=32, n_heads=4,
        rank=16, rff_dim=64, k_max=32, stride=4,
        lsh_buckets=64, lsh_bandwidth=2.0, lsh_key_dim=32,
        wu_manber_prefix_len=4, dropout=0.1, intermediate_dim=256,
        use_rff=True, hyper_cuts_dim_groups=None, n_lsh_hashes=4
    )
    logger = create_logger("./logs")
    agent = SPPO(training_config, attention_config)
    previous_loss = float('inf')
    def dummy_reward_model(states, actions):
        return -torch.sum(actions**2, dim=-1, keepdim=True)
    logger.info("Training started")
    try:
        total_steps = 0
        episodes = training_config.total_timesteps // training_config.num_samples
        for episode in range(1, episodes + 1):
            agent.train_mode()
            agent.buffer.reset()
            for step in range(training_config.num_samples):
                state = torch.randn(training_config.num_samples, training_config.state_dim)
                norm_state = agent.normalize_state(state)
                action, _, _ = agent.sample_action(norm_state)
                reward = dummy_reward_model(norm_state, action)
                agent.buffer.add(norm_state, action, reward)
                total_steps += 1
                if total_steps % training_config.log_frequency == 0:
                    logger.info(f"Episode {episode}, Step {step}, TotalSteps {total_steps}")
            states_batch, actions_batch, rewards_batch = agent.buffer.get()
            loss = agent.compute_ppo_loss(states_batch, actions_batch, rewards_batch)
            agent.update_policy(loss)
            agent.update_baseline()
            current_loss = loss.item()
            agent.adjust_hyperparameters(current_loss, previous_loss)
            previous_loss = current_loss
            logger.info(f"Episode {episode} completed, Loss: {current_loss:.4f}, eta: {agent.eta:.3f}, alpha: {agent.alpha:.3f}")
            if total_steps % training_config.eval_frequency == 0:
                logger.info(f"Evaluation checkpoint at step {total_steps}")
            if total_steps % training_config.save_frequency == 0:
                os.makedirs("./models", exist_ok=True)
                save_path = f"./models/checkpoint_step{total_steps}.pth"
                agent.save_model(save_path)
                logger.info(f"Model saved at {save_path}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        os.makedirs("./models", exist_ok=True)
        agent.save_model("./models/final_model.pth")
        logger.info("Training finished, final model saved.")

# ===============================
# APIサーバー (FastAPI)
# ===============================
def run_api():
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("FastAPIとuvicornが必要です。`pip install fastapi uvicorn` を実行してください。")
        return

    class StateRequest(BaseModel):
        state: List[float]

    app = FastAPI()
    training_config = TrainingConfig(state_dim=128, action_dim=2)
    attention_config = FastAttentionConfig(
        d_model=128, d_key=32, d_query=32, n_heads=4,
        rank=16, rff_dim=64, k_max=32, stride=4,
        lsh_buckets=64, lsh_bandwidth=2.0, lsh_key_dim=32,
        wu_manber_prefix_len=4, dropout=0.1, intermediate_dim=256,
        use_rff=True, hyper_cuts_dim_groups=None, n_lsh_hashes=4
    )
    agent = SPPO(training_config, attention_config)
    model_path = "./models/final_model.pth"
    if os.path.exists(model_path):
        agent.load_model(model_path)
    else:
        print(f"Warning: {model_path} が見つかりません。API起動前にトレーニングを実施してください。")
    agent.eval_mode()

    @app.post("/predict")
    def predict_action(request: StateRequest):
        state_tensor = torch.tensor(request.state, dtype=torch.float32)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        action, _, info = agent.sample_action(state_tensor, deterministic=True)
        return {
            "action": action.cpu().numpy().tolist(),
            "value": info["value"].cpu().numpy().tolist()
        }

    uvicorn.run(app, host="0.0.0.0", port=8000)

# ===============================
# エントリーポイント
# ===============================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api()
    else:
        train_agent()
