"""
Fair Partition - A Python package for fair partitioning algorithms.
"""

# Import and expose the main classes
from .partition_estimation import FairGroups, FairKMeans
from .fairness_metrics import compute_phi_sp_ci, get_conditional_positive_y_proba

__version__ = "0.1.0"

# Make modules available for direct import
from . import visualization

# Explicitly expose the classes and functions
__all__ = [
    "FairGroups",
    "FairKMeans",
    "compute_phi_sp_ci",
    "get_conditional_positive_y_proba",
    "visualization"
]
