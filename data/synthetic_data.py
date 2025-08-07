import numpy as np


def load_synthetic_data(n_groups=5, n_obs=10000):
    """
    Generate synthetic data for testing fair partitioning algorithms.
    
    This function creates a synthetic dataset with a continuous sensitive attribute
    and binary outcome variable. The data is designed to have a clear partition
    structure where different ranges of the sensitive attribute have different
    probabilities of positive outcomes, making it suitable for testing fairness
    partitioning algorithms like FairGroups and FairKMeans.
    
    Parameters
    ----------
    n_groups : int, default=5
        Number of groups in the ground truth partition. This determines how many
        distinct regions the sensitive attribute is divided into.
    n_obs : int, default=10000
        Number of observations per group. The total dataset size will be
        n_groups * n_obs.
        
    Returns
    -------
    s : ndarray of shape (n_groups * n_obs,)
        Continuous sensitive attribute values uniformly distributed between 0 and 100.
        This represents the feature that will be partitioned by the fairness algorithms.
    y : ndarray of shape (n_groups * n_obs,)
        Binary outcome variable (0 or 1) with different probabilities for each group.
        The probability of positive outcome increases linearly across groups.
    gt_partition : ndarray of shape (n_groups + 1,)
        Ground truth partition boundaries that define the optimal grouping of the
        sensitive attribute. These boundaries are used for evaluation purposes.
    y_probs : ndarray of shape (n_groups,)
        True probabilities of positive outcome for each group in the ground truth
        partition. These probabilities increase linearly from 0.1 to 0.9.
        
    Notes
    -----
    The synthetic data is generated as follows:
    1. Sensitive attribute 's' is uniformly distributed between 0 and 100
    2. Ground truth partition divides the range [0, 100] into n_groups regions
    3. Each region has a different probability of positive outcome (y=1)
    4. Probabilities increase linearly from 0.1 (first group) to 0.9 (last group)
    5. Binary outcomes are generated using binomial distribution with these probabilities
    
    This creates a dataset where fairness partitioning algorithms should ideally
    recover the ground truth partition boundaries to maximize fairness variance.
    """
    # Generate synthetic data
    n_groups = 5  # Number of groups to partition into
    n_obs = 10000  # Number of observations per group

    # Define ground truth partition boundaries
    gt_partition = np.array([0, 20, 30, 55, 88, 100])

    # Generate random feature values L uniformly distributed between 0 and 100
    # This is a sensitive variable
    s_min = 0  # Minimum value for feature L
    s_max = 100  # Maximum value for feature L
    s = np.random.uniform(0, 100, n_obs*n_groups)

    # Generate binary labels Y with different probabilities for each group
    y_probs = np.linspace(0.1, 0.9, n_groups)  # Linearly spaced probabilities from 0.1 to 0.9
    y = np.zeros(len(s))

    # Assign labels based on the ground truth partition
    for i in range(len(gt_partition)-1):
        mask = (s >= gt_partition[i]) & (s <= gt_partition[i+1])
        y[mask] = np.random.binomial(1, y_probs[i], np.sum(mask))

    return s, y, gt_partition, y_probs