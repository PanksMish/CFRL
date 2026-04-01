"""dataset package — data loading, partitioning, and crowdsource simulation."""
from .data_loader import FakeNewsDataset, load_and_split_data, get_dataloaders
from .federated_partitioner import FederatedPartitioner
from .crowdsource_simulator import CrowdsourceSimulator

__all__ = [
    "FakeNewsDataset",
    "load_and_split_data",
    "get_dataloaders",
    "FederatedPartitioner",
    "CrowdsourceSimulator",
]
