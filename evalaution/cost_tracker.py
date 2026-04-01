"""
evaluation/cost_tracker.py
---------------------------
Tracks training and communication costs across federated rounds.

The paper's system-level cost analysis (Figure 3) shows that CFRL-FND
reduces total cost to ~104 normalised units vs. ~125–136 for baselines.

Cost model (§5.2):
    - Communication cost baseline: DistilBERT ≈ 250 MB/round
    - Training cost: wall-clock time (normalised)
    - Total cost metric: training_cost + communication_cost

Cost reduction achieved by CFRL-FND: ~20% vs. FedAvg/SCAFFOLD.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Records per-round training and communication costs.

    Args:
        num_nodes : Total number of virtual nodes.
        base_mb   : Baseline communication cost in MB (DistilBERT = 250 MB).
    """

    def __init__(
        self,
        num_nodes: int  = cfg.federated.DEFAULT_NODES,
        base_mb:   float = cfg.federated.COMM_COST_MB_PER_ROUND,
    ):
        self.num_nodes = num_nodes
        self.base_mb   = base_mb

        # Per-round history
        self._round_training_costs:  List[float] = []
        self._round_comm_costs:      List[float] = []
        self._round_selected_counts: List[int]   = []
        self._round_times:           List[float] = []

        self._round_start: Optional[float] = None

    def start_round(self) -> None:
        """Mark the beginning of a communication round."""
        self._round_start = time.perf_counter()

    def end_round(
        self,
        selected_node_ids:   List[int],
        node_training_costs: Dict[int, float],
        node_comm_costs:     Dict[int, float],
    ) -> Dict[str, float]:
        """
        Record costs for a completed round.

        Args:
            selected_node_ids   : IDs of nodes that participated.
            node_training_costs : Dict[node_id → training time (s)].
            node_comm_costs     : Dict[node_id → comm cost (norm. units)].

        Returns:
            Dict with round-level cost summary.
        """
        elapsed = 0.0
        if self._round_start is not None:
            elapsed           = time.perf_counter() - self._round_start
            self._round_start = None

        n_selected    = len(selected_node_ids)
        avg_train     = float(np.mean(list(node_training_costs.values()))) if node_training_costs else 0.0
        avg_comm      = float(np.mean(list(node_comm_costs.values())))     if node_comm_costs     else 0.0

        # Normalise total cost to arbitrary units (a.u.) where CEN-FND = 100
        # Communication cost scales with fraction of nodes participating
        participation_fraction = n_selected / max(self.num_nodes, 1)
        norm_comm = participation_fraction * 100.0    # 100 a.u. = full participation
        norm_train = avg_train / 60.0 * 100.0          # 100 a.u. = 60 seconds

        self._round_training_costs.append(avg_train)
        self._round_comm_costs.append(avg_comm)
        self._round_selected_counts.append(n_selected)
        self._round_times.append(elapsed)

        summary = {
            "training_cost_sec":   avg_train,
            "comm_cost_norm":      avg_comm,
            "selected_nodes":      n_selected,
            "participation_frac":  participation_fraction,
            "norm_train_au":       norm_train,
            "norm_comm_au":        norm_comm,
            "total_norm_au":       norm_train + norm_comm,
            "elapsed_sec":         elapsed,
        }

        logger.debug(
            "Round cost — train=%.2fs | comm=%.1f a.u. | selected=%d/%d",
            avg_train, avg_comm, n_selected, self.num_nodes,
        )
        return summary

    def total_communication_cost(self) -> float:
        """
        Compute total communication cost across all rounds.

        Formula: num_rounds × participation_rate × base_mb (MB)
        """
        if not self._round_selected_counts:
            return 0.0
        avg_selected  = np.mean(self._round_selected_counts)
        num_rounds    = len(self._round_selected_counts)
        total_mb      = num_rounds * (avg_selected / self.num_nodes) * self.base_mb
        return float(total_mb)

    def total_training_cost_seconds(self) -> float:
        """Total wall-clock training time across all rounds."""
        return float(sum(self._round_training_costs))

    def cost_reduction_vs_baseline(
        self,
        baseline_total_cost: float = 123.0,  # FedAvg normalised cost (Fig. 3)
    ) -> float:
        """
        Compute cost reduction percentage vs. a baseline method.

        Default baseline is FedAvg (normalised cost ≈ 123 a.u. from Fig. 3).

        Returns:
            Percentage reduction ∈ [0, 100].
        """
        if not self._round_training_costs:
            return 0.0
        our_cost   = float(np.mean(self._round_training_costs) + np.mean(self._round_comm_costs))
        reduction  = (baseline_total_cost - our_cost) / baseline_total_cost * 100.0
        return float(np.clip(reduction, 0.0, 100.0))

    def get_cost_history(self) -> Dict[str, List[float]]:
        """Return full cost history for plotting."""
        return {
            "training_cost":   self._round_training_costs,
            "comm_cost":       self._round_comm_costs,
            "selected_counts": [float(x) for x in self._round_selected_counts],
            "round_times":     self._round_times,
        }

    def summary(self) -> Dict[str, float]:
        """Return cost summary statistics."""
        if not self._round_training_costs:
            return {}
        return {
            "total_training_sec":    self.total_training_cost_seconds(),
            "total_comm_mb":         self.total_communication_cost(),
            "mean_selected":         float(np.mean(self._round_selected_counts)),
            "mean_participation_pct": float(np.mean(self._round_selected_counts) / self.num_nodes * 100),
            "num_rounds":            len(self._round_training_costs),
        }
