"""
models/federated_node.py
-------------------------
Virtual federated node — the fundamental participant in FED-RL (§4.4).

Each virtual node represents a logical data silo (e.g., a social-media
community cluster). It maintains:
    - A local copy of the global model
    - Local training data (from the federated partitioner)
    - Cost metrics: training cost, communication cost, data quality
    - A similarity score δ_v with respect to the query article

Local training follows Equation 18:
    ω_v_t = ω_G_t - γ · ∇F_v(ω_G_t)

Communication cost is approximated from parameter count (§5.2):
    DistilBERT ≈ 66M parameters × 4 bytes = 250 MB per round
"""

import copy
import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import cfg

logger = logging.getLogger(__name__)


class FederatedNode:
    """
    Represents a single virtual node in the federated learning network.

    Each node maintains a local model copy and trains on its private data.
    After local training, only the model *updates* (gradients / parameters)
    are sent to the aggregator — raw data never leaves the node.

    Args:
        node_id        : Unique integer identifier for this node.
        model          : The global model (deep-copied for local use).
        dataloader     : Local DataLoader for this node's private data.
        device         : Torch device for training.
        local_epochs   : Number of local training epochs per round.
        learning_rate  : Learning rate for local optimiser.
        credibility    : Crowdsourcing credibility score ∈ [0, 1].
        similarity     : Similarity index δ_v with the query article.
    """

    def __init__(
        self,
        node_id:       int,
        model:         nn.Module,
        dataloader:    DataLoader,
        device:        torch.device,
        local_epochs:  int   = cfg.federated.LOCAL_EPOCHS,
        learning_rate: float = cfg.train.LEARNING_RATE,
        credibility:   float = 0.8,
        similarity:    float = 0.5,
    ):
        self.node_id      = node_id
        self.device       = device
        self.dataloader   = dataloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        # Local model copy (stays on the node — never shared directly)
        self.local_model  = copy.deepcopy(model).to(device)

        # Crowdsourcing credibility and similarity scores
        self.credibility  = credibility
        self.similarity   = similarity     # δ_v

        # Cost tracking
        self.training_cost     = 0.0
        self.communication_cost = 0.0
        self.diffusion_cost    = 0.0

        # Sample count for weighted aggregation
        self.num_samples  = len(dataloader.dataset)

    def synchronise(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Download the global model parameters (ω_G_t) to this node.

        This simulates the communication overhead of broadcasting the global
        model at the start of each round.

        Args:
            global_state_dict : Global model's state dictionary.
        """
        self.local_model.load_state_dict(copy.deepcopy(global_state_dict))

    def local_train(self) -> Tuple[float, float]:
        """
        Perform local training for *local_epochs* epochs on private data.

        Implements Equation 18:
            ω_v_t = ω_G_t - γ · ∇F_v(ω_G_t)

        Returns:
            (avg_loss, elapsed_time_seconds) after all local epochs.
        """
        self.local_model.train()
        optimizer  = optim.AdamW(
            self.local_model.parameters(),
            lr=self.learning_rate,
            weight_decay=cfg.train.WEIGHT_DECAY,
        )
        criterion  = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        t_start    = time.perf_counter()

        for epoch in range(self.local_epochs):
            for batch in self.dataloader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits, probs, _ = self.local_model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.local_model.parameters(),
                    max_norm=cfg.train.GRAD_CLIP,
                )
                optimizer.step()

                total_loss  += loss.item()
                num_batches += 1

        elapsed = time.perf_counter() - t_start
        avg_loss = total_loss / max(num_batches, 1)

        # Record training cost (normalised wall-clock time)
        self.training_cost = elapsed
        logger.debug(
            "Node %d | epochs=%d | avg_loss=%.4f | time=%.2fs",
            self.node_id, self.local_epochs, avg_loss, elapsed,
        )

        return avg_loss, elapsed

    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """
        Return the local model's state dictionary after training.

        In a real FL system only parameter *deltas* (gradients) would be sent.
        Here we return the full state dict, which the aggregator uses to
        compute the weighted average (Equation 19).

        Returns:
            Local model state dictionary (CPU tensors for efficient transfer).
        """
        return {k: v.cpu().clone() for k, v in self.local_model.state_dict().items()}

    def compute_communication_cost(
        self,
        bytes_per_param: int = 4,
        base_mb: float       = cfg.federated.COMM_COST_MB_PER_ROUND,
    ) -> float:
        """
        Estimate communication cost in normalised units.

        The paper uses DistilBERT ≈ 250 MB per round as baseline (§5.2).
        The actual cost scales with the fraction of selected nodes.

        Returns:
            Communication cost in normalised arbitrary units (a.u.).
        """
        num_params       = sum(p.numel() for p in self.local_model.parameters())
        approx_mb        = (num_params * bytes_per_param) / (1024 ** 2)
        # Normalise relative to baseline
        self.communication_cost = approx_mb / base_mb * 100.0
        return self.communication_cost

    def compute_quality_cost(self, global_loss: float) -> float:
        """
        Compute quality-aware cost component C_Q^t (Equation 7):

            C_Q = Σ_v (exp(δ_v) + 1) · Σ_j Loss(ω_v)

        A node with low similarity (δ_v ≈ 0) contributes less quality penalty
        because its local data is less relevant to the global objective.

        Args:
            global_loss : Current global validation loss (proxy for Σ_j Loss(ω_v)).

        Returns:
            Quality-aware cost contribution for this node.
        """
        quality_multiplier = np.exp(self.similarity) + 1.0
        cost               = quality_multiplier * global_loss
        return float(cost)

    @property
    def state_vector(self) -> np.ndarray:
        """
        Build the RL state vector for this node (used by the DDPG agent):
            [training_cost (norm), comm_cost (norm), diffusion_cost (norm),
             data_quality, similarity, num_samples (norm)]

        All values normalised to [0, 1] for stable RL training.
        """
        return np.array([
            min(self.training_cost / 60.0, 1.0),         # normalise by 60s
            min(self.communication_cost / 100.0, 1.0),
            min(self.diffusion_cost / 10.0, 1.0),
            self.credibility,                             # already ∈ [0,1]
            self.similarity,                              # already ∈ [0,1]
            min(self.num_samples / 1000.0, 1.0),
        ], dtype=np.float32)
