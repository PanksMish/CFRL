"""
training/reward_calculator.py
------------------------------
Implements the reward function and cost components for the DDPG agent.

Reward function (Equation 4):
    R_t = α · ΔAcc_t − β · C_t

Total cost (Equation 5):
    C_t = C_T^t + C_Q^t

Training + communication cost (Equation 6):
    C_T^t = Σ_v∈V (c_l^(t,v) + c_c^(t,v) + c_d^(t,v)) / |V|

Quality-aware cost (Equation 7):
    C_Q^t = Σ_v∈V (exp(δ_v) + 1) · Σ_j Loss(ω_v)

where:
    c_l = local training cost (wall-clock time, normalised)
    c_c = communication cost  (MB transferred, normalised)
    c_d = diffusion cost      (proportional to node connectivity)
    δ_v = similarity index    (Equation 15)
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Computes the reward signal and cost breakdown for the DDPG agent.

    The reward balances accuracy gain against system cost, encouraging the
    agent to learn node selection policies that improve accuracy while
    reducing communication and training overhead.

    Args:
        alpha  : Accuracy improvement weight (α in Eq. 4).
        beta   : Cost penalty weight          (β in Eq. 4).
    """

    def __init__(
        self,
        alpha: float = cfg.rl.ALPHA,
        beta:  float = cfg.rl.BETA,
    ):
        self.alpha = alpha
        self.beta  = beta

        # History for computing Δacc (accuracy improvement)
        self._prev_accuracy: Optional[float] = None

    def compute_reward(
        self,
        current_accuracy:   float,
        selected_nodes:     List[int],
        node_training_costs: Dict[int, float],
        node_comm_costs:    Dict[int, float],
        node_diff_costs:    Dict[int, float],
        node_similarities:  Dict[int, float],
        global_loss:        float,
    ) -> Dict[str, float]:
        """
        Compute the reward R_t and its components.

        Args:
            current_accuracy     : Validation accuracy at round t.
            selected_nodes       : List of selected node IDs.
            node_training_costs  : Dict[node_id → training cost (seconds)].
            node_comm_costs      : Dict[node_id → comm cost (norm. units)].
            node_diff_costs      : Dict[node_id → diffusion cost].
            node_similarities    : Dict[node_id → δ_v similarity score].
            global_loss          : Current global validation loss.

        Returns:
            Dict with keys: 'reward', 'delta_acc', 'cost_T', 'cost_Q', 'cost_total'
        """
        # ── Accuracy Improvement ΔAcc_t ──────────────────────────────────────
        if self._prev_accuracy is None:
            delta_acc = 0.0   # First round: no reference
        else:
            delta_acc = current_accuracy - self._prev_accuracy
        self._prev_accuracy = current_accuracy

        # ── C_T^t: Training + Communication + Diffusion Cost ─────────────────
        cost_T = self._compute_system_cost(
            selected_nodes,
            node_training_costs,
            node_comm_costs,
            node_diff_costs,
        )

        # ── C_Q^t: Quality-Aware Penalty ─────────────────────────────────────
        cost_Q = self._compute_quality_cost(
            selected_nodes,
            node_similarities,
            global_loss,
        )

        # ── Total Cost ────────────────────────────────────────────────────────
        cost_total = cost_T + cost_Q

        # ── Reward R_t = α · ΔAcc_t − β · C_t ───────────────────────────────
        reward = self.alpha * delta_acc - self.beta * cost_total

        result = {
            "reward":      float(reward),
            "delta_acc":   float(delta_acc),
            "cost_T":      float(cost_T),
            "cost_Q":      float(cost_Q),
            "cost_total":  float(cost_total),
        }

        logger.debug(
            "Reward: %.4f | ΔAcc: %.4f | C_T: %.4f | C_Q: %.4f",
            reward, delta_acc, cost_T, cost_Q,
        )
        return result

    def _compute_system_cost(
        self,
        selected_nodes:     List[int],
        node_training_costs: Dict[int, float],
        node_comm_costs:    Dict[int, float],
        node_diff_costs:    Dict[int, float],
    ) -> float:
        """
        Compute C_T^t = Σ_v∈V* (c_l + c_c + c_d) / |V*|  (Equation 6).

        All cost components are normalised to [0, 1] before aggregation.
        """
        if not selected_nodes:
            return 0.0

        total = 0.0
        for v in selected_nodes:
            c_l = node_training_costs.get(v, 0.0)
            c_c = node_comm_costs.get(v, 0.0)
            c_d = node_diff_costs.get(v, 0.0)

            # Normalise each cost to [0, 1]
            c_l_norm = min(c_l / 60.0, 1.0)       # Normalise by 60 seconds
            c_c_norm = min(c_c / 100.0, 1.0)       # Normalise by 100 units
            c_d_norm = min(c_d / 10.0, 1.0)        # Normalise by 10 units

            total += c_l_norm + c_c_norm + c_d_norm

        return total / len(selected_nodes)

    def _compute_quality_cost(
        self,
        selected_nodes:    List[int],
        node_similarities: Dict[int, float],
        global_loss:       float,
    ) -> float:
        """
        Compute C_Q^t = Σ_v∈V* (exp(δ_v) + 1) · Loss(ω_v)  (Equation 7).

        The loss term acts as a proxy for the local model's quality gap.
        Nodes with high similarity (δ_v → 1) incur a higher quality weight,
        incentivising selection of nodes most relevant to the query.
        """
        if not selected_nodes:
            return 0.0

        cost_Q = 0.0
        for v in selected_nodes:
            delta_v  = node_similarities.get(v, 0.5)
            weight   = np.exp(delta_v) + 1.0        # exp(δ_v) + 1
            cost_Q  += weight * global_loss

        # Normalise by number of selected nodes and exp range
        normaliser = (np.exp(1.0) + 1.0) * len(selected_nodes)
        return cost_Q / normaliser if normaliser > 0 else 0.0

    def reset(self) -> None:
        """Reset accuracy history (call at the start of a new experiment)."""
        self._prev_accuracy = None


def compute_global_rl_state(
    selected_nodes:      List[int],
    node_training_costs: Dict[int, float],
    node_comm_costs:     Dict[int, float],
    node_diff_costs:     Dict[int, float],
    node_similarities:   Dict[int, float],
    node_qualities:      Dict[int, float],
    round_num:           int,
    max_rounds:          int,
) -> np.ndarray:
    """
    Build the global RL state vector s_t for the DDPG agent.

    The state aggregates node-level statistics into a compact 6-D vector:
        s_t = [mean_train_cost, mean_comm_cost, mean_diff_cost,
               mean_quality, mean_similarity, round_progress]

    All values normalised to [0, 1].

    Args:
        selected_nodes       : Currently selected node IDs (for computing context).
        node_training_costs  : Dict[node_id → training cost].
        node_comm_costs      : Dict[node_id → comm cost].
        node_diff_costs      : Dict[node_id → diffusion cost].
        node_similarities    : Dict[node_id → similarity score δ_v].
        node_qualities       : Dict[node_id → data quality / credibility].
        round_num            : Current communication round (0-indexed).
        max_rounds           : Total planned communication rounds.

    Returns:
        state : (STATE_DIM,) numpy float32 array.
    """
    all_nodes = list(node_training_costs.keys())

    def _safe_mean(d: Dict, keys: List[int]) -> float:
        vals = [d.get(k, 0.0) for k in keys]
        return float(np.mean(vals)) if vals else 0.0

    mean_train   = _safe_mean(node_training_costs, all_nodes)
    mean_comm    = _safe_mean(node_comm_costs,     all_nodes)
    mean_diff    = _safe_mean(node_diff_costs,     all_nodes)
    mean_quality = _safe_mean(node_qualities,      all_nodes)
    mean_sim     = _safe_mean(node_similarities,   all_nodes)
    round_prog   = round_num / max(max_rounds - 1, 1)

    state = np.array([
        min(mean_train   / 60.0,  1.0),
        min(mean_comm    / 100.0, 1.0),
        min(mean_diff    / 10.0,  1.0),
        float(np.clip(mean_quality, 0.0, 1.0)),
        float(np.clip(mean_sim,    0.0, 1.0)),
        float(np.clip(round_prog,  0.0, 1.0)),
    ], dtype=np.float32)

    return state
