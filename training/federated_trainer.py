"""
training/federated_trainer.py
------------------------------
Main Federated Reinforcement Learning training loop — Algorithm 2 (FED-RL).

Implements the CFRL-FND end-to-end training workflow (Algorithm 3):
    1. QDO: Preprocess input
    2. IES: Extract features and similarity scores
    3. DRLSelect: Select optimal node subset V*
    4. FederatedTraining: Local training + global aggregation
    5. Score aggregation and evaluation

Federated aggregation (Equation 19):
    ω_G_{t+1} = Σ_{v∈V*} (|P_v| / Σ_{u∈V*}|P_u|) · ω_v_t

The training loop interleaves FL rounds with RL updates, allowing the DDPG
agent to learn node selection policies that balance accuracy and cost.
"""

import copy
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from dataset.federated_partitioner import FederatedPartitioner
from dataset.crowdsource_simulator import CrowdsourceSimulator
from evaluation.metrics import compute_metrics
from evaluation.cost_tracker import CostTracker
from models.information_extraction import InformationExtractionSubsystem
from models.federated_node import FederatedNode
from models.score_aggregator import ScoreAggregator
from training.node_selector import NodeSelector
from training.reward_calculator import RewardCalculator, compute_global_rl_state
from utils.logger import get_logger

logger = get_logger(__name__)


class FederatedTrainer:
    """
    Orchestrates the full CFRL-FND training procedure.

    Manages:
        - Global model state
        - Virtual node pool (FederatedNode instances)
        - DDPG-based node selector
        - Reward computation and RL updates
        - Federated aggregation
        - Evaluation across rounds

    Args:
        global_model   : The global IES model to be federated.
        partitioner    : Non-IID data partitioner.
        val_loader     : Validation DataLoader for round-level evaluation.
        test_loader    : Test DataLoader for final evaluation.
        num_nodes      : Number of virtual nodes.
        num_rounds     : Total communication rounds.
        device         : Torch device.
        crowd_sim      : Crowdsource simulator (optional).
        save_dir       : Directory to save checkpoints.
    """

    def __init__(
        self,
        global_model:  InformationExtractionSubsystem,
        partitioner:   FederatedPartitioner,
        val_loader:    DataLoader,
        test_loader:   DataLoader,
        num_nodes:     int = cfg.federated.DEFAULT_NODES,
        num_rounds:    int = cfg.federated.NUM_ROUNDS,
        device:        Optional[torch.device] = None,
        crowd_sim:     Optional[CrowdsourceSimulator] = None,
        save_dir:      str = cfg.paths.CHECKPOINTS_DIR,
    ):
        self.global_model = global_model
        self.partitioner  = partitioner
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.num_nodes    = num_nodes
        self.num_rounds   = num_rounds
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crowd_sim    = crowd_sim or CrowdsourceSimulator(num_nodes=num_nodes)
        self.save_dir     = save_dir

        self.global_model.to(self.device)

        # Node pool: initialised lazily per round (to avoid memory overload)
        self._node_pool: Optional[Dict[int, FederatedNode]] = None

        # RL components
        self.node_selector   = NodeSelector(num_nodes=num_nodes)
        self.reward_calc     = RewardCalculator()
        self.score_aggregator = ScoreAggregator()

        # Cost tracker
        self.cost_tracker = CostTracker(num_nodes=num_nodes)

        # Training history
        self.history = {
            "round_acc":        [],   # Val accuracy per round
            "round_loss":       [],   # Val loss per round
            "selected_count":   [],   # # nodes selected per round
            "reward":           [],   # RL reward per round
            "comm_cost":        [],   # Communication cost per round
            "train_cost":       [],   # Training cost per round
            "critic_loss":      [],   # DDPG critic loss
            "actor_loss":       [],   # DDPG actor loss
        }

        logger.info(
            "FederatedTrainer: %d nodes | %d rounds | device=%s",
            num_nodes, num_rounds, self.device,
        )

    def _build_node_pool(self) -> Dict[int, FederatedNode]:
        """
        Instantiate all virtual nodes with their local data loaders.
        Called once before training begins.
        """
        logger.info("Building virtual node pool (%d nodes)...", self.num_nodes)
        node_pool = {}

        for v in range(self.num_nodes):
            loader      = self.partitioner.get_node_dataloader(v, batch_size=cfg.train.BATCH_SIZE)
            credibility = self.crowd_sim.get_node_credibility(v)
            similarity  = np.random.uniform(0.3, 0.9)  # Will be updated per article

            node = FederatedNode(
                node_id      = v,
                model        = self.global_model,
                dataloader   = loader,
                device       = self.device,
                local_epochs = cfg.federated.LOCAL_EPOCHS,
                credibility  = credibility,
                similarity   = similarity,
            )
            node_pool[v] = node

        logger.info("Node pool ready.")
        return node_pool

    def _aggregate_global_model(
        self,
        selected_nodes: List[int],
        node_updates:   Dict[int, Dict[str, torch.Tensor]],
    ) -> None:
        """
        Federated weighted aggregation (Equation 19):
            ω_G_{t+1} = Σ_{v∈V*} (|P_v| / Σ_{u∈V*}|P_u|) · ω_v_t

        Updates the global model's state dictionary in-place.

        Args:
            selected_nodes : List of participating node IDs.
            node_updates   : Dict[node_id → local model state dict].
        """
        # Compute normalised weights based on local dataset sizes
        total_samples = sum(
            self._node_pool[v].num_samples for v in selected_nodes
        )
        weights = {
            v: self._node_pool[v].num_samples / total_samples
            for v in selected_nodes
        }

        # Initialise aggregated state dict with zeros
        global_sd  = self.global_model.state_dict()
        agg_sd     = {k: torch.zeros_like(v) for k, v in global_sd.items()}

        for v in selected_nodes:
            w      = weights[v]
            node_sd = node_updates[v]
            for k in agg_sd:
                if k in node_sd:
                    agg_sd[k] += w * node_sd[k].to(self.device)

        self.global_model.load_state_dict(agg_sd)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the global model on a DataLoader.

        Returns:
            (accuracy, avg_loss)
        """
        self.global_model.eval()
        criterion  = nn.CrossEntropyLoss()
        all_preds, all_labels = [], []
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"].to(self.device)

            logits, probs, _ = self.global_model(input_ids, attention_mask)
            loss             = criterion(logits, labels)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            total_loss += loss.item()
            n_batches  += 1

        metrics  = compute_metrics(np.array(all_labels), np.array(all_preds))
        avg_loss = total_loss / max(n_batches, 1)

        self.global_model.train()
        return metrics["accuracy"], avg_loss

    def train(self) -> Dict:
        """
        Execute the full CFRL-FND federated training loop.

        Each round:
            1. Broadcast global model to all nodes (simulate sync).
            2. NodeSelector (DDPG) selects V* ⊂ V.
            3. Selected nodes run local_train().
            4. Global model aggregated from local updates.
            5. Evaluate on validation set.
            6. Compute reward and update DDPG agent.
            7. Log metrics and costs.

        Returns:
            Training history dictionary.
        """
        if self._node_pool is None:
            self._node_pool = self._build_node_pool()

        logger.info("=" * 60)
        logger.info("Starting CFRL-FND Federated Training")
        logger.info("Rounds: %d | Nodes: %d", self.num_rounds, self.num_nodes)
        logger.info("=" * 60)

        # Initialise RL state
        node_qualities = self.node_selector.compute_node_quality_scores(
            node_credibilities={v: self._node_pool[v].credibility for v in range(self.num_nodes)},
            node_similarities={v: self._node_pool[v].similarity   for v in range(self.num_nodes)},
        )

        # Initial state vector
        rl_state = compute_global_rl_state(
            selected_nodes      = list(range(self.num_nodes)),
            node_training_costs = {v: 0.0 for v in range(self.num_nodes)},
            node_comm_costs     = {v: 0.0 for v in range(self.num_nodes)},
            node_diff_costs     = {v: 0.0 for v in range(self.num_nodes)},
            node_similarities   = {v: self._node_pool[v].similarity for v in range(self.num_nodes)},
            node_qualities      = node_qualities,
            round_num           = 0,
            max_rounds          = self.num_rounds,
        )

        best_val_acc   = 0.0
        best_model_sd  = None
        prev_action    = np.array([cfg.federated.PARTICIPATION_RATE], dtype=np.float32)

        for rnd in tqdm(range(self.num_rounds), desc="FL Rounds", unit="round"):
            logger.info("\n--- Round %d / %d ---", rnd + 1, self.num_rounds)

            # ── Step 1: Node Selection ────────────────────────────────────────
            selected_nodes, threshold = self.node_selector.select_nodes(
                state          = rl_state,
                node_qualities = node_qualities,
                explore        = True,
            )
            self.history["selected_count"].append(len(selected_nodes))
            logger.info("Selected %d nodes (threshold=%.3f)", len(selected_nodes), threshold)

            # ── Step 2: Synchronise Global Model to Selected Nodes ────────────
            global_sd = self.global_model.state_dict()
            for v in selected_nodes:
                self._node_pool[v].synchronise(global_sd)

            # ── Step 3: Local Training ────────────────────────────────────────
            node_updates       = {}
            node_training_costs = {}
            node_comm_costs    = {}

            for v in selected_nodes:
                node = self._node_pool[v]
                avg_loss, elapsed = node.local_train()

                node_updates[v]        = node.get_model_update()
                node_training_costs[v] = elapsed
                node_comm_costs[v]     = node.compute_communication_cost()

            # ── Step 4: Federated Aggregation ─────────────────────────────────
            self._aggregate_global_model(selected_nodes, node_updates)

            # ── Step 5: Evaluation ────────────────────────────────────────────
            val_acc, val_loss = self._evaluate(self.val_loader)
            self.history["round_acc"].append(val_acc)
            self.history["round_loss"].append(val_loss)

            logger.info(
                "Round %d | val_acc=%.4f | val_loss=%.4f | nodes=%d",
                rnd + 1, val_acc, val_loss, len(selected_nodes),
            )

            # ── Step 6: Reward Computation ────────────────────────────────────
            reward_info = self.reward_calc.compute_reward(
                current_accuracy    = val_acc,
                selected_nodes      = selected_nodes,
                node_training_costs = node_training_costs,
                node_comm_costs     = node_comm_costs,
                node_diff_costs     = {v: 0.1 for v in selected_nodes},
                node_similarities   = {v: self._node_pool[v].similarity for v in range(self.num_nodes)},
                global_loss         = val_loss,
            )
            self.history["reward"].append(reward_info["reward"])
            self.history["comm_cost"].append(
                float(np.mean(list(node_comm_costs.values()))) if node_comm_costs else 0.0
            )
            self.history["train_cost"].append(
                float(np.mean(list(node_training_costs.values()))) if node_training_costs else 0.0
            )

            # ── Step 7: Build Next RL State ───────────────────────────────────
            next_rl_state = compute_global_rl_state(
                selected_nodes      = selected_nodes,
                node_training_costs = node_training_costs,
                node_comm_costs     = node_comm_costs,
                node_diff_costs     = {v: 0.1 for v in selected_nodes},
                node_similarities   = {v: self._node_pool[v].similarity for v in range(self.num_nodes)},
                node_qualities      = node_qualities,
                round_num           = rnd + 1,
                max_rounds          = self.num_rounds,
            )

            # ── Step 8: DDPG Update ───────────────────────────────────────────
            action_taken = np.array([threshold], dtype=np.float32)
            c_loss, a_loss = self.node_selector.update_agent(
                state      = rl_state,
                action     = action_taken,
                reward     = reward_info["reward"],
                next_state = next_rl_state,
                done       = (rnd == self.num_rounds - 1),
            )
            if c_loss is not None:
                self.history["critic_loss"].append(c_loss)
                self.history["actor_loss"].append(a_loss)

            rl_state    = next_rl_state
            prev_action = action_taken

            # ── Track Best Model ──────────────────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_model_sd = copy.deepcopy(self.global_model.state_dict())
                logger.info("New best val_acc=%.4f at round %d", best_val_acc, rnd + 1)

        # ── Restore Best Model and Final Test Evaluation ──────────────────────
        if best_model_sd is not None:
            self.global_model.load_state_dict(best_model_sd)

        logger.info("=" * 60)
        logger.info("Training complete. Best val_acc=%.4f", best_val_acc)

        # Save best model
        ckpt_path = os.path.join(self.save_dir, "cfrl_fnd_best.pt")
        torch.save(best_model_sd, ckpt_path)
        logger.info("Best model saved to %s", ckpt_path)

        return self.history

    def evaluate_test(self) -> Dict:
        """
        Evaluate the (best) global model on the held-out test set.

        Returns:
            Full metrics dictionary (accuracy, precision, recall, F1, ROC-AUC).
        """
        self.global_model.eval()
        all_preds, all_labels, all_probs = [], [], []
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                logits, probs, _ = self.global_model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs[:, 1].cpu().numpy().tolist())
                total_loss += loss.item()

        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )
        metrics["avg_loss"] = total_loss / max(len(self.test_loader), 1)

        logger.info("=" * 60)
        logger.info("Test Results:")
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)
        logger.info("=" * 60)

        return metrics
