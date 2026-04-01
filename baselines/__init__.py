"""baselines package — FedAvg, SCAFFOLD, and Centralized baselines."""
from .fed_avg import FedAvgTrainer
from .scaffold import ScaffoldTrainer
from .centralized import CentralizedTrainer

__all__ = ["FedAvgTrainer", "ScaffoldTrainer", "CentralizedTrainer"]
