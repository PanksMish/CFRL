"""
baselines/fed_avg.py
---------------------
FedAvg Baseline — McMahan et al. (2017), §5.3

Communication-Efficient Learning of Deep Networks from Decentralized Data.
FedAvg is the canonical federated learning algorithm: at each round, a random
subset of clients trains locally, and the server aggregates by weighted average.

Key difference from CFRL-FND:
    - Node selection is RANDOM (not RL-based)
    - No cost-awareness or quality weighting
    - All selected nodes participate with equal probability

Paper results (Figure 2): FedAvg achieves ~90.4% accuracy on Dataset A,
which is 3-5% below CFRL-FND due to static client selection.
"""

import copy
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from dataset.federated_partitioner import FederatedPartitioner
from evaluation.metrics import compute_metrics
from models.information_extraction import InformationExtractionSubsystem
from utils.logger import get_logger

logger = get_logger(__name__)


class FedAvgTrainer:
    """
    FedAvg federated learning trainer (Algorithm 1 of McMahan et al., 2017).

    At each round:
        1. Select K random clients (participation_rate × num_nodes).
        2. Broadcast global model to selected clients.
        3. Each client performs local SGD for E epochs.
        4. Server aggregates: ω_G ← Σ_k (n_k/n) · ω_k

    Args:
        global_model      : Global IES model.
        partitioner       : Non-IID data partitioner.
        val_loader        : Validation DataLoader.
        test_loader       : Test DataLoader.
        num_nodes         : Total number of virtual nodes.
        num_rounds        : Communication rounds.
        participation_rate: Fraction of nodes selected per round.
        device            : Torch device.
        save_dir          : Checkpoint directory.
    """

    def __init__(
        self,
        global_model:       InformationExtractionSubsystem,
        partitioner:        FederatedPartitioner,
        val_loader:         DataLoader,
        test_loader:        DataLoader,
        num_nodes:          int   = cfg.federated.DEFAULT_NODES,
        num_rounds:         int   = cfg.federated.NUM_ROUNDS,
        participation_rate: float = cfg.federated.PARTICIPATION_RATE,
        device:             Optional[torch.device] = None,
        save_dir:           str = cfg.paths.CHECKPOINTS_DIR,
    ):
        self.global_model       = global_model
        self.partitioner        = partitioner
        self.val_loader         = val_loader
        self.test_loader        = test_loader
        self.num_nodes          = num_nodes
        self.num_rounds         = num_rounds
        self.k                  = max(1, int(num_nodes * participation_rate))
        self.device             = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir           = save_dir

        self.global_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.history: Dict[str, List[float]] = {
            "round_acc": [], "round_loss": [], "comm_cost": [], "train_cost": []
        }
        logger.info(
            "FedAvgTrainer: %d nodes | K=%d | %d rounds",
            num_nodes, self.k, num_rounds,
        )

    def _local_train(
        self,
        node_id:    int,
        model:      nn.Module,
        dataloader: DataLoader,
    ) -> Tuple[Dict, float, float]:
        """
        Local training on a single node.

        Returns:
            (state_dict, avg_loss, num_samples)
        """
        model.train()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.train.LEARNING_RATE,
            weight_decay=cfg.train.WEIGHT_DECAY,
        )

        total_loss, n_batches = 0.0, 0
        for _epoch in range(cfg.federated.LOCAL_EPOCHS):
            for batch in dataloader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits, _, _ = model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.GRAD_CLIP)
                optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

        avg_loss   = total_loss / max(n_batches, 1)
        num_samples = self.partitioner.get_node_sample_count(node_id)
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}, avg_loss, float(num_samples)

    def _aggregate(
        self,
        node_updates:   Dict[int, Dict],
        node_weights:   Dict[int, float],
    ) -> None:
        """FedAvg aggregation: ω_G ← Σ_k w_k · ω_k."""
        global_sd = self.global_model.state_dict()
        agg_sd    = {k: torch.zeros_like(v) for k, v in global_sd.items()}

        for node_id, update in node_updates.items():
            w = node_weights[node_id]
            for k in agg_sd:
                if k in update:
                    agg_sd[k] += w * update[k].to(self.device)

        self.global_model.load_state_dict(agg_sd)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate global model. Returns (accuracy, avg_loss)."""
        self.global_model.eval()
        all_preds, all_labels = [], []
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"].to(self.device)

            logits, _, _ = self.global_model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches  += 1

        acc = compute_metrics(np.array(all_labels), np.array(all_preds))["accuracy"]
        return acc, total_loss / max(n_batches, 1)

    def train(self) -> Dict:
        """
        Run FedAvg for num_rounds rounds.

        Returns:
            Training history dictionary.
        """
        logger.info("Starting FedAvg training (%d nodes, %d rounds) ...", self.num_nodes, self.num_rounds)
        best_val_acc  = 0.0
        best_model_sd = None

        for rnd in tqdm(range(self.num_rounds), desc="FedAvg Rounds", unit="round"):
            # ── 1. Random client selection ────────────────────────────────────
            selected = np.random.choice(self.num_nodes, size=self.k, replace=False).tolist()

            # ── 2. Local training ─────────────────────────────────────────────
            node_updates  = {}
            node_n_samples = {}
            global_sd     = self.global_model.state_dict()

            for v in selected:
                local_model = copy.deepcopy(self.global_model)
                loader      = self.partitioner.get_node_dataloader(v, batch_size=cfg.train.BATCH_SIZE)
                update, _, n = self._local_train(v, local_model, loader)
                node_updates[v]   = update
                node_n_samples[v] = n

            # ── 3. Weighted aggregation ───────────────────────────────────────
            total_n    = sum(node_n_samples.values())
            node_weights = {v: n / total_n for v, n in node_n_samples.items()}
            self._aggregate(node_updates, node_weights)

            # ── 4. Evaluate ───────────────────────────────────────────────────
            val_acc, val_loss = self._evaluate(self.val_loader)
            self.history["round_acc"].append(val_acc)
            self.history["round_loss"].append(val_loss)
            # Communication cost: full model × K nodes (no reduction)
            self.history["comm_cost"].append(float(self.k / self.num_nodes * 100.0))

            logger.info(
                "FedAvg Round %d | val_acc=%.4f | K=%d",
                rnd + 1, val_acc, self.k,
            )

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_model_sd = copy.deepcopy(self.global_model.state_dict())

        if best_model_sd:
            self.global_model.load_state_dict(best_model_sd)
            ckpt = os.path.join(self.save_dir, "fedavg_best.pt")
            torch.save(best_model_sd, ckpt)

        return self.history

    def evaluate_test(self) -> Dict:
        """Evaluate on held-out test set."""
        acc, _ = self._evaluate(self.test_loader)
        logger.info("FedAvg Test Accuracy: %.4f", acc)
        return {"accuracy": acc}
