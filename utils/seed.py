"""
utils/seed.py
--------------
Sets global random seeds for full reproducibility across:
    - Python  random module
    - NumPy   random number generator
    - PyTorch CPU and GPU operations
    - CUDA    deterministic algorithms (with cuDNN)
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducible experiments.

    Args:
        seed : Integer seed value (default: 42, matching cfg.data.RANDOM_SEED).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN operations (may reduce speed slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    logger.info("Global random seed set to %d.", seed)
