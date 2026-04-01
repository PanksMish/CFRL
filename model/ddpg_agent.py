"""
models/ddpg_agent.py
---------------------
Deep Deterministic Policy Gradient (DDPG) agent for cost-aware node selection
in the FED-RL module (§3.3, §4.4).

The DDPG agent learns an adaptive policy π* that selects the optimal subset
of virtual nodes V* at each communication round to:
    - Maximise prediction accuracy improvement (ΔAcc_t)
    - Minimise total system cost C_t (training + communication + quality penalty)

Architecture:
    Actor  : State → continuous action (participation threshold ∈ [0, 1])
    Critic : (State, Action) → expected Q-value (scalar)

Both networks use soft target updates with τ = 0.005 for training stability.

State vector s_t (dimension = 6):
    [training_cost, comm_cost, diffusion_cost, data_quality, avg_similarity, round_num]

Action a_t (dimension = 1):
    Continuous threshold θ ∈ [0, 1]. Nodes with quality score > θ are selected.
"""

import copy
import logging
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Experience replay buffer for off-policy DDPG training.

    Stores transitions (s, a, r, s', done) and samples random mini-batches
    for training, breaking temporal correlations between consecutive samples.

    Args:
        capacity  : Maximum number of transitions to store.
        state_dim : Dimensionality of the state vector.
        action_dim: Dimensionality of the action vector.
    """

    def __init__(
        self,
        capacity:   int = cfg.rl.REPLAY_BUFFER_SIZE,
        state_dim:  int = cfg.rl.STATE_DIM,
        action_dim: int = cfg.rl.ACTION_DIM,
    ):
        self.capacity   = capacity
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # Pre-allocate arrays for efficiency
        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

        self.ptr  = 0       # Write pointer
        self.size = 0       # Current occupancy

    def push(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Store a single transition."""
        i = self.ptr % self.capacity
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random mini-batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors.
        """
        idx = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_states[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def __len__(self) -> int:
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
# Actor Network
# ─────────────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Policy network: maps state → deterministic action.

    Architecture:
        Linear(state_dim, 256) → LayerNorm → ReLU →
        Linear(256, 128)       → LayerNorm → ReLU →
        Linear(128, action_dim)→ Sigmoid (outputs ∈ [0, 1])

    The sigmoid output represents the participation threshold θ.
    Nodes with normalised quality scores > θ are selected for the round.
    """

    def __init__(
        self,
        state_dim:  int = cfg.rl.STATE_DIM,
        action_dim: int = cfg.rl.ACTION_DIM,
        hidden1:    int = 256,
        hidden2:    int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
            nn.Sigmoid(),                  # Output ∈ [0, 1]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise final layer with small weights for stable starts."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state : (B, state_dim) normalised state tensor.
        Returns:
            action : (B, action_dim) participation threshold.
        """
        return self.net(state)


# ─────────────────────────────────────────────────────────────────────────────
# Critic Network
# ─────────────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Value network: maps (state, action) → Q-value scalar.

    Architecture:
        Concatenate(state, action) →
        Linear(state_dim + action_dim, 256) → LayerNorm → ReLU →
        Linear(256, 128)                    → LayerNorm → ReLU →
        Linear(128, 1)
    """

    def __init__(
        self,
        state_dim:  int = cfg.rl.STATE_DIM,
        action_dim: int = cfg.rl.ACTION_DIM,
        hidden1:    int = 256,
        hidden2:    int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        state:  torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state  : (B, state_dim)
            action : (B, action_dim)
        Returns:
            q_value : (B, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# DDPG Agent
# ─────────────────────────────────────────────────────────────────────────────

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent for node selection.

    Implements the policy described in §3.3 of CFRL-FND:
        - Markov Decision Process for node selection
        - Actor-Critic with soft target updates
        - Gaussian exploration noise (§5.2: N(0, σ²) with σ=0.2)
        - Experience replay for off-policy learning

    The agent's learned policy converges to a locally optimal π* by
    minimising the Bellman error (Critic) and following the deterministic
    policy gradient theorem (Actor), as proven in Theorem 3.1.

    Args:
        state_dim    : State vector dimensionality.
        action_dim   : Action vector dimensionality.
        device       : Torch device.
    """

    def __init__(
        self,
        state_dim:  int           = cfg.rl.STATE_DIM,
        action_dim: int           = cfg.rl.ACTION_DIM,
        device:     torch.device  = None,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Networks ──────────────────────────────────────────────────────────
        self.actor        = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic        = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # ── Optimisers ────────────────────────────────────────────────────────
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=cfg.rl.ACTOR_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.rl.CRITIC_LR)

        # ── Replay Buffer ─────────────────────────────────────────────────────
        self.replay_buffer = ReplayBuffer(
            capacity=cfg.rl.REPLAY_BUFFER_SIZE,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        # ── Hyperparameters ───────────────────────────────────────────────────
        self.gamma     = cfg.rl.GAMMA
        self.tau       = cfg.rl.TAU
        self.noise_std = cfg.rl.NOISE_STD
        self.batch_rl  = cfg.rl.BATCH_SIZE_RL

        # Training step counter (used to track warmup)
        self.step_count   = 0
        self.warmup_steps = cfg.rl.WARMUP_STEPS_RL

        logger.info(
            "DDPGAgent on %s | state_dim=%d, action_dim=%d, γ=%.3f, τ=%.4f",
            self.device, state_dim, action_dim, self.gamma, self.tau,
        )

    def select_action(
        self,
        state:  np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:
        """
        Select an action given the current state.

        During exploration (explore=True), Gaussian noise N(0, σ²) is added
        to the deterministic actor output to encourage exploration.

        Args:
            state   : (state_dim,) numpy array.
            explore : Whether to add exploration noise.

        Returns:
            action : (action_dim,) numpy array clipped to [0, 1].
        """
        if self.step_count < self.warmup_steps:
            # Random exploration during warmup
            return np.random.uniform(0.0, 1.0, size=(self.action_dim,)).astype(np.float32)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        self.actor.train()

        if explore:
            noise  = np.random.normal(0.0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, 0.0, 1.0)

        return action.astype(np.float32)

    def store_transition(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Push a transition into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

    def update(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Perform one DDPG update step.

        1. Sample mini-batch from replay buffer.
        2. Update Critic by minimising TD error (Bellman equation).
        3. Update Actor by maximising Q-value (policy gradient).
        4. Soft-update target networks.

        Returns:
            (critic_loss, actor_loss) or (None, None) if buffer not yet warm.
        """
        if len(self.replay_buffer) < self.batch_rl:
            return None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_rl, self.device
        )

        # ── Critic Update (minimise Bellman error) ────────────────────────────
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_q       = self.critic_target(next_states, target_actions)
            y              = rewards + self.gamma * (1.0 - dones) * target_q

        current_q    = self.critic(states, actions)
        critic_loss  = F.mse_loss(current_q, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # ── Actor Update (maximise Q-value via policy gradient) ───────────────
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        # ── Soft Target Update ────────────────────────────────────────────────
        self._soft_update(self.actor_target,  self.actor)
        self._soft_update(self.critic_target, self.critic)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """
        θ_target ← τ·θ_source + (1 - τ)·θ_target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str) -> None:
        """Save agent state dictionaries."""
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "step_count":    self.step_count,
        }, path)
        logger.info("DDPGAgent saved to %s", path)

    def load(self, path: str) -> None:
        """Load agent state dictionaries."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.step_count = ckpt.get("step_count", 0)
        logger.info("DDPGAgent loaded from %s", path)


# Fix missing Optional import
from typing import Optional
