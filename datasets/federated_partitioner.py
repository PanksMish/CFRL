"""
dataset/federated_partitioner.py
---------------------------------
Implements non-IID data partitioning for federated learning simulation.

The CFRL-FND paper simulates 50–500 virtual nodes with non-IID data
distributions (§5.2). This module uses a Dirichlet distribution (α = 0.5)
to create heterogeneous local datasets that mirror real-world federated
settings where different social-media communities have skewed class ratios.

Non-IID Dirichlet partitioning (standard in FL literature):
  - For each class c, sample a Dirichlet vector of proportions across nodes
  - Assign samples accordingly so that each node has a different class mix
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import cfg

logger = logging.getLogger(__name__)


class FederatedPartitioner:
    """
    Partitions a dataset into non-IID shards for virtual federated nodes.

    Attributes:
        dataset      : The base PyTorch Dataset to partition.
        num_nodes    : Number of virtual nodes.
        alpha        : Dirichlet concentration parameter.
                       Lower α → more heterogeneous (skewed) distributions.
        seed         : Random seed for reproducibility.

    Usage:
        partitioner = FederatedPartitioner(train_dataset, num_nodes=100)
        node_loaders = partitioner.get_node_dataloaders(batch_size=32)
    """

    def __init__(
        self,
        dataset,
        num_nodes: int = cfg.federated.DEFAULT_NODES,
        alpha: float   = cfg.federated.NON_IID_ALPHA,
        seed: int      = cfg.data.RANDOM_SEED,
    ):
        self.dataset   = dataset
        self.num_nodes = num_nodes
        self.alpha     = alpha
        self.rng       = np.random.default_rng(seed)

        # Map from node_id → list of global sample indices
        self._node_indices: Dict[int, List[int]] = {}
        self._partition()

    def _partition(self) -> None:
        """
        Perform Dirichlet non-IID partitioning.

        For each class, draw a Dirichlet(α) vector of size num_nodes,
        multiply by the number of class samples, and assign them to nodes.
        This creates local datasets with varying class ratios.
        """
        labels = np.array([self.dataset[i]["label"].item() for i in range(len(self.dataset))])
        num_classes = len(np.unique(labels))

        # Collect indices per class
        class_indices: Dict[int, np.ndarray] = {}
        for c in range(num_classes):
            idx = np.where(labels == c)[0]
            self.rng.shuffle(idx)
            class_indices[c] = idx

        # Initialise empty index lists for each node
        node_indices: Dict[int, List[int]] = {v: [] for v in range(self.num_nodes)}

        for c in range(num_classes):
            idx    = class_indices[c]
            n      = len(idx)

            # Dirichlet proportions for this class across all nodes
            props  = self.rng.dirichlet(np.repeat(self.alpha, self.num_nodes))
            # Convert proportions to integer counts (ensure sum == n)
            counts = (props * n).astype(int)
            counts[-1] += n - counts.sum()   # Remainder to last node

            start = 0
            for v, cnt in enumerate(counts):
                end = start + cnt
                node_indices[v].extend(idx[start:end].tolist())
                start = end

        self._node_indices = node_indices

        sizes = [len(v) for v in node_indices.values()]
        logger.info(
            "Non-IID partitioning complete (α=%.2f): "
            "min_samples=%d | max_samples=%d | mean_samples=%.1f",
            self.alpha, min(sizes), max(sizes), np.mean(sizes),
        )

    def get_node_indices(self, node_id: int) -> List[int]:
        """Return the list of global sample indices assigned to a node."""
        if node_id not in self._node_indices:
            raise ValueError(f"node_id {node_id} not in range [0, {self.num_nodes})")
        return self._node_indices[node_id]

    def get_node_dataloader(
        self,
        node_id: int,
        batch_size: int = cfg.train.BATCH_SIZE,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Build a DataLoader for a single virtual node.

        Args:
            node_id     : Index of the virtual node.
            batch_size  : Mini-batch size.
            num_workers : DataLoader worker processes.

        Returns:
            DataLoader containing only this node's local data.
        """
        indices = self.get_node_indices(node_id)
        subset  = Subset(self.dataset, indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(len(subset) > batch_size),
        )

    def get_node_dataloaders(
        self,
        batch_size: int = cfg.train.BATCH_SIZE,
        num_workers: int = 0,
    ) -> Dict[int, DataLoader]:
        """
        Build DataLoaders for ALL virtual nodes.

        Returns:
            Dict mapping node_id → DataLoader.
        """
        loaders = {}
        for v in range(self.num_nodes):
            loaders[v] = self.get_node_dataloader(v, batch_size, num_workers)
        logger.info("Created %d node DataLoaders.", self.num_nodes)
        return loaders

    def get_node_sample_count(self, node_id: int) -> int:
        """Return number of samples at the given node."""
        return len(self._node_indices[node_id])

    def get_total_samples(self) -> int:
        """Return total samples across all nodes (should equal dataset size)."""
        return sum(len(v) for v in self._node_indices.values())

    @property
    def node_weights(self) -> Dict[int, float]:
        """
        Compute the fractional weight of each node (|P_v| / |P|), used
        during federated aggregation (Equation 3 in the paper).
        """
        total = self.get_total_samples()
        return {v: len(idx) / total for v, idx in self._node_indices.items()}

    def get_node_class_distribution(self, node_id: int) -> Dict[int, int]:
        """Return class-count dict for a given node (for analysis)."""
        indices = self.get_node_indices(node_id)
        dist: Dict[int, int] = {}
        for i in indices:
            lbl = self.dataset[i]["label"].item()
            dist[lbl] = dist.get(lbl, 0) + 1
        return dist
