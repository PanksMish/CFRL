"""
baselines/centralized.py
-------------------------
Centralized Learning Baseline (CEN-FND) — §5.3

Standard centralized training where all data is aggregated and the model
is trained on a single server. This represents the upper bound on accuracy
but violates data privacy assumptions of federated learning.

CEN-FND serves as the accuracy ceiling reference in Figure 2 and Figure 11.
The paper reports CEN-FND achieves ~87.3% accuracy on Dataset A, which is
lower than CFRL-FND (95.08%) because it does not benefit from the diverse
non-IID data distributions across federated nodes.
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
from evaluation.metrics import compute_metrics
from models.information_extraction import InformationExtractionSubsystem
from utils.logger import get_logger

logger = get_logger(__name__)


class CentralizedTrainer:
    """
    Standard centralized training without federated constraints.

    Trains the IES model on a single aggregated dataset with full data
    access. Used as a non-private baseline for accuracy comparison.

    Args:
        model        : IES model to train.
        train_loader : Training DataLoader (full aggregated dataset).
        val_loader   : Validation DataLoader.
        test_loader  : Test DataLoader.
        device       : Torch device.
        max_epochs   : Maximum training epochs.
        save_dir     : Directory for checkpoints.
    """

    def __init__(
        self,
        model:        InformationExtractionSubsystem,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        test_loader:  DataLoader,
        device:       Optional[torch.device] = None,
        max_epochs:   int = cfg.train.MAX_EPOCHS,
        save_dir:     str = cfg.paths.CHECKPOINTS_DIR,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs   = max_epochs
        self.save_dir     = save_dir

        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.LEARNING_RATE,
            weight_decay=cfg.train.WEIGHT_DECAY,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=1e-6
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []
        }

    def _train_epoch(self) -> float:
        """Run one full training epoch. Returns average training loss."""
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for batch in self.train_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits, _, _ = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.GRAD_CLIP)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate model. Returns (val_loss, accuracy, f1)."""
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss, n_batches = 0.0, 0

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"].to(self.device)

            logits, _, _ = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches  += 1

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, metrics["accuracy"], metrics["f1"]

    def train(self) -> Dict:
        """
        Run full centralized training with early stopping.

        Returns:
            Training history dictionary.
        """
        logger.info("Starting CEN-FND (Centralized) training for %d epochs ...", self.max_epochs)
        best_val_acc  = 0.0
        best_model_sd = None
        patience_ctr  = 0

        for epoch in tqdm(range(self.max_epochs), desc="CEN-FND Epochs", unit="epoch"):
            train_loss = self._train_epoch()
            val_loss, val_acc, val_f1 = self._evaluate(self.val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            logger.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f",
                epoch + 1, train_loss, val_loss, val_acc, val_f1,
            )

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_model_sd = copy.deepcopy(self.model.state_dict())
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.train.PATIENCE:
                    logger.info("Early stopping at epoch %d (patience=%d).", epoch + 1, cfg.train.PATIENCE)
                    break

        # Restore best model
        if best_model_sd:
            self.model.load_state_dict(best_model_sd)

        ckpt = os.path.join(self.save_dir, "cen_fnd_best.pt")
        torch.save(best_model_sd, ckpt)
        logger.info("CEN-FND best model saved to %s", ckpt)

        return self.history

    def evaluate_test(self) -> Dict:
        """Evaluate on the held-out test set."""
        _, accuracy, f1 = self._evaluate(self.test_loader)
        metrics = {"accuracy": accuracy, "f1": f1}
        logger.info("CEN-FND Test — acc=%.4f | f1=%.4f", accuracy, f1)
        return metrics
