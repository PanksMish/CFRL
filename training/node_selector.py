"""
training/node_selector.py
--------------------------
Cost-Aware Node Selection (DRLSelect) — Algorithm 1 from the paper (§4.4).

The NodeSelector wraps the DDPG agent and translates the continuous
action output (participation threshold θ) into a discrete selection
of virtual nodes for each communication round.

Selection logic:
    1. DDPG agent observes global state s_t → outputs threshold θ ∈ [0, 1].
    2. Each node has a quality score q_v = credibility × (1 + similarity) / 2.
    3. Nodes with q_v ≥ θ are selected as V* for the current round.
    4. Fallback: if too few nodes selected, revert to top-k by quality score.

This dynamic approach outperforms heuristic baselines (random, top-k)
by learning long-term trade-offs between node quality and system cost.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import cfg
from models.ddpg_agent import DDPGAgent

logger = logging.getLogger(__name__)


class NodeSelector:
    """
    Wraps the DDPG agent to provide cost-aware virtual node selection.

    Implements Algorithm 1 (DRLSelect) from the paper:
        Input  : Node set V, similarity scores {δ_v}
        Output : Selected node subset V*

    The selector manages the full RL loop:
        - State construction
        - Action selection (threshold θ via DDPG)
        - Node filtering based on quality scores
        - Experience storage after observing the next state and reward

    Args:
        num_nodes         : Total number of virtual nodes.
        min_selected      : Minimum nodes to select per round (floor).
        max_selected      : Maximum nodes to select per round (ceiling).
        agent             : Pre-initialised DDPGAgent (creates one if None).
        participation_rate: Fraction of total nodes to target per round.
    """

    def __init__(
        self,
        num_nodes:          int,
        min_selected:       Optional[int] = None,
        max_selected:       Optional[int] = None,
        agent:              Optional[DDPGAgent] = None,
        participation_rate: float = cfg.federated.PARTICIPATION_RATE,
    ):
        self.num_nodes     = num_nodes
        self.target_k      = max(1, int(num_nodes * participation_rate))
        self.min_selected  = min_selected or max(1, self.target_k // 2)
        self.max_selected  = max_selected or min(num_nodes, self.target_k * 2)
        self.agent         = agent or DDPGAgent()

        logger.info(
            "NodeSelector: %d total nodes | target_k=%d | min=%d | max=%d",
            num_nodes, self.target_k, self.min_selected, self.max_selected,
        )

    def compute_node_quality_scores(
        self,
        node_credibilities: Dict[int, float],
        node_similarities:  Dict[int, float],
    ) -> Dict[int, float]:
        """
        Compute a composite quality score for each node.

        Quality combines credibility (crowdsource reliability) and similarity
        (semantic relevance to the query article):
            q_v = (credibility_v + similarity_v) / 2

        Args:
            node_credibilities : Dict[node_id → credibility ∈ [0, 1]].
            node_similarities  : Dict[node_id → similarity δ_v ∈ [0, 1]].

        Returns:
            Dict[node_id → quality score ∈ [0, 1]].
        """
        quality = {}
        for v in range(self.num_nodes):
            cred = node_credibilities.get(v, 0.5)
            sim  = node_similarities.get(v, 0.5)
            quality[v] = (cred + sim) / 2.0
        return quality

    def select_nodes(
        self,
        state:              np.ndarray,
        node_qualities:     Dict[int, float],
        explore:            bool = True,
    ) -> Tuple[List[int], float]:
        """
        Select a subset of nodes using the DDPG policy.

        The agent outputs threshold θ; nodes with quality > θ are selected.
        If selection is too small, fall back to top-k by quality.

        Args:
            state          : (STATE_DIM,) RL state vector.
            node_qualities : Dict[node_id → quality score].
            explore        : Whether to add exploration noise to the action.

        Returns:
            (selected_node_ids, threshold_used)
        """
        # Agent selects threshold θ
        action    = self.agent.select_action(state, explore=explore)
        threshold = float(action[0])

        # Select nodes with quality score ≥ threshold
        selected = [
            v for v, q in node_qualities.items()
            if q >= threshold
        ]

        # Enforce min/max constraints
        if len(selected) < self.min_selected:
            # Not enough nodes — take top-k by quality score
            sorted_nodes = sorted(node_qualities, key=node_qualities.get, reverse=True)
            selected     = sorted_nodes[:self.target_k]
            logger.debug(
                "Fallback to top-%d selection (threshold=%.3f gave only %d nodes).",
                self.target_k, threshold, len(selected),
            )
        elif len(selected) > self.max_selected:
            # Too many nodes — subsample to max_selected
            selected = sorted(selected, key=lambda v: node_qualities[v], reverse=True)
            selected = selected[:self.max_selected]

        logger.debug(
            "Selected %d/%d nodes | threshold=%.3f",
            len(selected), self.num_nodes, threshold,
        )
        return selected, threshold

    def update_agent(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Store transition and perform one DDPG update step.

        Args:
            state      : State at time t.
            action     : Action taken at time t.
            reward     : Reward received after taking action.
            next_state : State at time t+1.
            done       : Whether the episode terminated.

        Returns:
            (critic_loss, actor_loss) from the DDPG update.
        """
        self.agent.store_transition(state, action, reward, next_state, done)
        return self.agent.update()

    def random_select(self, k: Optional[int] = None) -> List[int]:
        """
        Select k random nodes (used as baseline comparison).

        Args:
            k : Number of nodes to select (defaults to target_k).

        Returns:
            List of randomly selected node IDs.
        """
        k = k or self.target_k
        return list(np.random.choice(self.num_nodes, size=min(k, self.num_nodes), replace=False))

    def heuristic_select(
        self,
        node_qualities: Dict[int, float],
        k: Optional[int] = None,
    ) -> List[int]:
        """
        Greedy top-k selection by quality score (heuristic baseline).

        Args:
            node_qualities : Dict[node_id → quality score].
            k              : Number of nodes to select.

        Returns:
            List of top-k node IDs sorted by quality (descending).
        """
        k       = k or self.target_k
        sorted_ = sorted(node_qualities, key=node_qualities.get, reverse=True)
        return sorted_[:k]
