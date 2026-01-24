"""Seeding utilities for reproducibility."""

import random
import numpy as np
import os


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Seeds Python random, NumPy, and PyTorch (if available).
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to seed PyTorch if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Try to seed TensorFlow if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
