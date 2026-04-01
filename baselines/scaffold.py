"""
baselines/scaffold.py
---------------------
SCAFFOLD Baseline — Karimireddy et al. (2020), §5.3

Stochastic Controlled Averaging for Federated Learning.
SCAFFOLD uses control variates to correct for client drift in heterogeneous
(non-IID) settings. At each round, clients update both the local model and
a local control variate c_i, sending updates Δy_i and Δc_i to the server.

Server update:
    x ← x + η_g / S · Σ_i Δy_i
    c ← c + 1/N · Σ_i Δc_i

Client update:
    y_i ← y_i - η_l · (∇F_i(y_i) - c_i + c)

Key advantage over FedAvg: corrects client-drift in non-IID data,
leading to faster convergence. Paper reports ~91.2% accuracy on Dataset A.
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


class ScaffoldTrainer:
    """
    SCAFFOLD federated learning trainer.

    Implements client-drift correction via control variates.
    Each client i maintains a control variate c_i that estimates the
    gradient direction correction needed for the global objective.

    Args:
        global_model      : Global IES model.
        partitioner       : Non-IID data partitioner.
        val_loader        : Validation DataLoader.
        test_loader       : Test DataLoader.
        num_nodes         : Total virtual nodes.
        num_rounds        : Communication rounds.
        participation_rate: Fraction of nodes per round.
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

        # Initialise global and per-client control variates
        self.global_c:  Dict[str, torch.Tensor] = self._zero_like_model()
        self.client_c:  Dict[int, Dict[str, torch.Tensor]] = {
            v: self._zero_like_model() for v in range(num_nodes)
        }

        self.history: Dict[str, List[float]] = {
            "round_acc": [], "round_loss": [], "comm_cost": []
        }
        logger.info(
            "ScaffoldTrainer: %d nodes | K=%d | %d rounds",
            num_nodes, self.k, num_rounds,
        )

    def _zero_like_model(self) -> Dict[str, torch.Tensor]:
        """Return zero tensors matching the global model's parameter shapes."""
        return {
            k: torch.zeros_like(v.cpu())
            for k, v in self.global_model.state_dict().items()
        }

    def _local_scaffold_train(
        self,
        node_id:    int,
        model:      nn.Module,
        dataloader: DataLoader,
    ) -> Tuple[Dict, Dict, Dict, float]:
        """
        SCAFFOLD local training step.

        Returns:
            (model_update, delta_c_i, new_c_i, avg_loss)
        """
        model.train()
        lr        = cfg.train.LEARNING_RATE
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        x_init_sd = copy.deepcopy(self.global_model.state_dict())  # x (global params)
        c_global  = self.global_c                                   # c  (global control)
        c_i       = self.client_c[node_id]                          # c_i (client control)

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

                # Apply SCAFFOLD correction: subtract c_i, add c (global)
                for name, param in model.named_parameters():
                    if param.grad is not None and name in c_global:
                        correction = (c_i[name].to(self.device) - c_global[name].to(self.device))
                        param.grad.data.add_(correction)

                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.GRAD_CLIP)
                optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Compute Δy_i = y_i - x (model update)
        y_i_sd  = model.state_dict()
        delta_y = {k: (y_i_sd[k].cpu() - x_init_sd[k].cpu()) for k in y_i_sd}

        # Update local control variate:
        # new_c_i = c_i - c + 1/(K·η) · (x - y_i)
        K        = cfg.federated.LOCAL_EPOCHS * len(dataloader)
        new_c_i  = {}
        delta_c  = {}
        for k_name in c_i:
            new_c_i[k_name] = (
                c_i[k_name]
                - c_global[k_name]
                + (1.0 / (K * lr)) * (-delta_y[k_name])
            )
            delta_c[k_name] = new_c_i[k_name] - c_i[k_name]

        return (
            {k: v.cpu().clone() for k, v in y_i_sd.items()},
            delta_c,
            new_c_i,
            avg_loss,
        )

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
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
        """Run SCAFFOLD for num_rounds rounds."""
        logger.info("Starting SCAFFOLD training (%d nodes, %d rounds)...", self.num_nodes, self.num_rounds)
        best_val_acc  = 0.0
        best_model_sd = None

        for rnd in tqdm(range(self.num_rounds), desc="SCAFFOLD Rounds", unit="round"):
            # Random client selection
            selected = np.random.choice(self.num_nodes, size=self.k, replace=False).tolist()

            # Local SCAFFOLD updates
            node_model_updates = {}
            node_delta_c       = {}
            node_n_samples     = {}
            global_sd          = self.global_model.state_dict()

            for v in selected:
                local_model = copy.deepcopy(self.global_model)
                loader      = self.partitioner.get_node_dataloader(v, batch_size=cfg.train.BATCH_SIZE)
                y_i_sd, delta_c, new_c_i, _ = self._local_scaffold_train(v, local_model, loader)

                node_model_updates[v] = y_i_sd
                node_delta_c[v]       = delta_c
                node_n_samples[v]     = self.partitioner.get_node_sample_count(v)

                # Update local control variate
                self.client_c[v] = new_c_i

            # Server aggregation of model updates
            total_n    = sum(node_n_samples.values())
            agg_sd     = {k: torch.zeros_like(v) for k, v in global_sd.items()}
            for v in selected:
                w = node_n_samples[v] / total_n
                for k in agg_sd:
                    if k in node_model_updates[v]:
                        agg_sd[k] += w * node_model_updates[v][k].to(self.device)
            self.global_model.load_state_dict(agg_sd)

            # Update global control variate: c ← c + (1/N) · Σ Δc_i
            for k_name in self.global_c:
                delta_sum = sum(node_delta_c[v][k_name] for v in selected if k_name in node_delta_c[v])
                self.global_c[k_name] += delta_sum / self.num_nodes

            # Evaluate
            val_acc, val_loss = self._evaluate(self.val_loader)
            self.history["round_acc"].append(val_acc)
            self.history["round_loss"].append(val_loss)
            self.history["comm_cost"].append(float(self.k / self.num_nodes * 100.0))

            logger.info("SCAFFOLD Round %d | val_acc=%.4f", rnd + 1, val_acc)

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_model_sd = copy.deepcopy(self.global_model.state_dict())

        if best_model_sd:
            self.global_model.load_state_dict(best_model_sd)
            ckpt = os.path.join(self.save_dir, "scaffold_best.pt")
            torch.save(best_model_sd, ckpt)

        return self.history

    def evaluate_test(self) -> Dict:
        acc, _ = self._evaluate(self.test_loader)
        logger.info("SCAFFOLD Test Accuracy: %.4f", acc)
        return {"accuracy": acc}
