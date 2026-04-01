"""
models/score_aggregator.py
---------------------------
Score Aggregator — §4.5

Combines the outputs of the IES (contextual reasoning) and FED-RL
(distributed federated learning) modules into a final prediction.

Final score (Equation 20):
    Score = w_F · S_F + w_I · S_I,     w_F + w_I = 1

Final classification (Equation 20 continued):
    R = I[Score > τ]

where τ is the decision threshold (default: 0.50).

The weighting parameters (w_F, w_I) are tuned on the validation set
to balance contextual reasoning and distributed learning contributions.
The ablation study (§6.4, Figure 12) shows that w_F = 1.0 (pure FED-RL)
achieves 95.0% accuracy, while w_F = 0.6 / w_I = 0.4 achieves the best
balance of accuracy and interpretability.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import cfg

logger = logging.getLogger(__name__)


class ScoreAggregator(nn.Module):
    """
    Weighted aggregation of IES and FED-RL prediction scores.

    Can operate in two modes:
        1. Fixed weights   : w_F and w_I are fixed constants (default).
        2. Learnable weights: w_F and w_I are learned via a small linear layer
                              trained end-to-end (set learnable=True).

    Args:
        w_fedrl    : Weight for FED-RL score (w_F).
        w_ies      : Weight for IES score    (w_I).
        threshold  : Decision threshold τ for classification.
        learnable  : If True, use a learnable weighted combination.
    """

    def __init__(
        self,
        w_fedrl:   float = cfg.aggregator.W_FEDRL,
        w_ies:     float = cfg.aggregator.W_IES,
        threshold: float = cfg.aggregator.THRESHOLD,
        learnable: bool  = False,
    ):
        super().__init__()

        # Validate weights
        if abs((w_fedrl + w_ies) - 1.0) > 1e-6:
            raise ValueError(
                f"Aggregator weights must sum to 1.0, got {w_fedrl + w_ies:.4f}"
            )

        self.threshold = threshold
        self.learnable = learnable

        if learnable:
            # Learnable weighted combination via softmax-parameterised logits
            self.weight_logits = nn.Parameter(
                torch.tensor([np.log(w_fedrl / w_ies)]),   # Initialise from prior
                requires_grad=True,
            )
            logger.info("ScoreAggregator: learnable weights mode.")
        else:
            # Register as buffers (not trained, but tracked in state_dict)
            self.register_buffer("w_fedrl", torch.tensor(w_fedrl))
            self.register_buffer("w_ies",   torch.tensor(w_ies))
            logger.info(
                "ScoreAggregator: fixed weights w_F=%.2f, w_I=%.2f, τ=%.2f",
                w_fedrl, w_ies, threshold,
            )

    def _get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (w_F, w_I) as tensors."""
        if self.learnable:
            # Sigmoid → w_F; 1 - w_F → w_I
            w_f = torch.sigmoid(self.weight_logits)
            w_i = 1.0 - w_f
            return w_f, w_i
        return self.w_fedrl, self.w_ies

    def forward(
        self,
        s_fedrl: torch.Tensor,
        s_ies:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the final prediction score and binary label.

        Args:
            s_fedrl : (B,) or (B, 1) FED-RL probability scores for Fake class.
            s_ies   : (B,) or (B, 1) IES probability scores for Fake class.

        Returns:
            score   : (B,) weighted combined score ∈ [0, 1].
            pred    : (B,) binary predictions {0, 1}.
        """
        s_fedrl = s_fedrl.squeeze(-1) if s_fedrl.dim() == 2 else s_fedrl
        s_ies   = s_ies.squeeze(-1)   if s_ies.dim()   == 2 else s_ies

        w_f, w_i = self._get_weights()

        # Equation 20: Score = w_F · S_F + w_I · S_I
        score = w_f * s_fedrl + w_i * s_ies

        # R = I[Score > τ]
        pred  = (score > self.threshold).long()

        return score, pred

    def aggregate_numpy(
        self,
        s_fedrl: float,
        s_ies:   float,
    ) -> Tuple[float, int]:
        """
        Lightweight numpy version for inference without GPU tensors.

        Args:
            s_fedrl : Scalar FED-RL Fake probability.
            s_ies   : Scalar IES Fake probability.

        Returns:
            (score, prediction) as Python scalars.
        """
        if self.learnable:
            w_f = float(torch.sigmoid(self.weight_logits).item())
            w_i = 1.0 - w_f
        else:
            w_f = float(self.w_fedrl.item())
            w_i = float(self.w_ies.item())

        score = w_f * s_fedrl + w_i * s_ies
        pred  = int(score > self.threshold)
        return score, pred

    def tune_weights(
        self,
        val_s_fedrl: np.ndarray,
        val_s_ies:   np.ndarray,
        val_labels:  np.ndarray,
        n_steps:     int = 20,
    ) -> Tuple[float, float, float]:
        """
        Grid-search for optimal (w_F, w_I, τ) on validation data.

        Searches over w_F ∈ {0.0, 0.05, ..., 1.0} and
        τ ∈ {0.3, 0.35, ..., 0.7}, returning the configuration that
        maximises validation accuracy.

        Args:
            val_s_fedrl : (N,) FED-RL fake-class scores on validation set.
            val_s_ies   : (N,) IES   fake-class scores on validation set.
            val_labels  : (N,) ground-truth binary labels.
            n_steps     : Number of grid steps for w_F search.

        Returns:
            (best_w_f, best_w_i, best_threshold)
        """
        from sklearn.metrics import accuracy_score

        best_acc   = -1.0
        best_w_f   = cfg.aggregator.W_FEDRL
        best_tau   = cfg.aggregator.THRESHOLD

        w_f_grid  = np.linspace(0.0, 1.0, n_steps + 1)
        tau_grid  = np.linspace(0.30, 0.70, 9)

        for w_f in w_f_grid:
            w_i = 1.0 - w_f
            scores = w_f * val_s_fedrl + w_i * val_s_ies
            for tau in tau_grid:
                preds = (scores > tau).astype(int)
                acc   = accuracy_score(val_labels, preds)
                if acc > best_acc:
                    best_acc  = acc
                    best_w_f  = float(w_f)
                    best_tau  = float(tau)

        best_w_i = 1.0 - best_w_f

        # Update internal weights (fixed mode only)
        if not self.learnable:
            self.w_fedrl.fill_(best_w_f)
            self.w_ies.fill_(best_w_i)
            self.threshold = best_tau

        logger.info(
            "ScoreAggregator tuned — w_F=%.2f, w_I=%.2f, τ=%.2f, val_acc=%.4f",
            best_w_f, best_w_i, best_tau, best_acc,
        )
        return best_w_f, best_w_i, best_tau
