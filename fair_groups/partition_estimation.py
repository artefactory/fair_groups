import itertools
from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from .fairness_metrics import compute_phi_on_grid, compute_phi_sp_ci


def _compute_weights(groups):
    """
    Compute the number of elements in each group defined by the partition.

    Parameters
    ----------
    groups : ndarray of shape (n_obs,)
        Group assignments for each observation.

    Returns
    -------
    weights : ndarray of shape (n_groups,)
        Array of counts for each group.
    """
    weights = np.array([np.sum(groups == i) for i in range(np.max(groups) + 1)])
    return weights


def _compute_phi_by_group(groups, y):
    """
    Compute the fairness metric (phi) estimate for each group defined by the partition.

    Parameters
    ----------
    groups : ndarray of shape (n_obs,)
        Group assignments for each observation.
    y : array-like
        Binary outcome variable.

    Returns
    -------
    phi_by_group : ndarray of shape (n_groups,)
        Fairness metric (phi) estimate for each group.
    """
    n_groups = np.max(groups) + 1
    phi_by_group = np.full(n_groups, np.nan)
    for i in range(n_groups):
        group = (groups == i)
        if np.any(group):
            phi_by_group[i] = y[group].mean()
    return phi_by_group


def _compute_phi_by_group_ci(groups, y):
    """
    Compute confidence intervals for the fairness metric (phi) estimate in each group defined by the partition.

    Parameters
    ----------
    groups : ndarray of shape (n_obs,)
        Group assignments for each observation.
    y : array-like
        Binary outcome variable.

    Returns
    -------
    phi_by_group_ci : ndarray of shape (n_groups, 3)
        Confidence intervals for each group (center, lower bound, upper bound).
    """
    phi_by_group_ci = [
        compute_phi_sp_ci(groups == i, y)
        for i in range(np.max(groups) + 1)
    ]
    return np.array(phi_by_group_ci)


def _compute_std(weights, phi_by_group):
    """
    Compute the weighted standard deviation of the fairness metric (phi) on all groups defined by the partition.

    Parameters
    ----------
    weights : ndarray of shape (n_groups,)
        Counts for each group.
    phi_by_group : ndarray of shape (n_groups,)
        Fairness metric (phi) estimate for each group.

    Returns
    -------
    std : float
        Weighted standard deviation.
    """
    sum_w = np.sum(weights)
    mean = np.sum(weights * phi_by_group) / sum_w
    variance = (phi_by_group - mean) ** 2
    variance = np.sum(weights * variance) / sum_w
    std = np.sqrt(max(variance, 0))
    return std


class FairPartitionBase(ABC):
    """
    Abstract base class for partitioning continuous sensitive attributes into groups.
    
    This class provides the foundation for different approaches to partition continuous
    sensitive attributes into discrete groups for fairness analysis. The goal is to
    maximize the variance of fairness criterion phi across the resulting groups.
    
    Parameters
    ----------
    n_groups : int
        Number of groups to partition the sensitive attribute into.
    grid_size : int, default=100
        Number of grid points used for discretization and optimization.
        
    Attributes
    ----------
    n_groups : int
        Number of groups for partitioning.
    grid_size : int
        Grid size for discretization.
    variance : float, optional
        Variance of fairness metric across groups (set after fitting).
    partition : list or ndarray, optional
        Partition boundaries defining the groups (set after fitting).
    weights : ndarray, optional
        Number of samples in each group (set after fitting).
    phi_by_group : ndarray, optional
        Fairness metric (phi) values for each group (set after fitting).
        
    Notes
    -----
    This is an abstract base class that must be subclassed. Subclasses must implement
    the abstract methods `fit`, `recompute_fairness_statistics`, and `print`.
    
    The fairness metric phi represents the difference between the probability of
    positive outcome for a group and the overall probability of positive outcome.
    """
    def __init__(self, n_groups, grid_size=100):
        self.n_groups = n_groups
        self.grid_size = grid_size
        self.variance = None
        self.partition = None
        self.weights = None
        self.phi_by_group = None

    def predict(self, s):
        """
        Compute group(s) for a given sensitive attribute value(s).
        
        Parameters
        ----------
        s : array-like
            Sensitive attribute values.
            
        Returns
        -------
        groups : array-like
            Group assignments for each sensitive attribute value.
        """
        if self.partition is None:
            raise ValueError("Model must be fitted before making predictions.")
        
        s = np.asarray(s)
        groups = np.zeros_like(s, dtype=int)
        
        for i in range(self.n_groups):
            mask = (s >= self.partition[i]) & (s <= self.partition[i + 1])
            groups[mask] = i
            
        return groups

    def recompute_fairness_statistics(self, s, y):
        """
        Compute fairness metric (phi) estimate, confidence intervals, and standard deviation for new data using the learned partition.

        Parameters
        ----------
        s : array-like
            Sensitive attribute values.
        y : array-like
            Binary outcome variable.

        Returns
        -------
        phi_by_group : ndarray
            Fairness metric (phi) estimate for each group.
        phi_by_group_ci : ndarray
            Confidence intervals for each group (center, lower bound, upper bound).
        std : float
            Weighted standard deviation.
        """
        groups = self.predict(s)
        phi_by_group = _compute_phi_by_group(groups, y)
        phi_by_group_ci = _compute_phi_by_group_ci(groups, y)
        weights = _compute_weights(groups)
        std = _compute_std(weights, phi_by_group)
        return phi_by_group, phi_by_group_ci, std

    def print(self):
        """Print information about the fitted partition."""
        if self.partition is None:
            print(f"{self.__class__.__name__} model is not fitted yet.")
            return
            
        print(f"{self.__class__.__name__} with {self.n_groups} groups")
        print(f"Partition boundaries: {self.partition}")
        if self.weights is not None:
            print(f"Group weights: {self.weights}")
        if self.phi_by_group is not None:
            print(f"Fairness metric (phi) by group: {self.phi_by_group}")
        if hasattr(self, 'std') and self.std is not None:
            print(f"Weighted standard deviation: {self.std:.6f}")

    @abstractmethod
    def fit(self, s, y):
        pass


class FairGroups(FairPartitionBase):
    """
    Partition continuous sensitive attribute into groups using exhaustive search.
    
    This implementation uses an exhaustive search approach to find the optimal partition
    that maximizes the variance of fairness criterion phi across groups. It evaluates
    all possible combinations of partition boundaries and selects the one that yields
    the highest weighted standard deviation of phi values.
    
    The algorithm works by:
    1. Creating a grid of possible partition points
    2. Evaluating all possible combinations of n_groups-1 internal partition points
    3. Computing phi values for each resulting partition
    4. Selecting the partition with maximum weighted standard deviation
    
    Parameters
    ----------
    n_groups : int
        Number of groups to partition the sensitive attribute into.
    grid_size : int, default=100
        Number of grid points for discretization and exhaustive search.
        
    Attributes
    ----------
    partition : list
        Partition boundaries defining the groups after fitting.
    phi_by_group : ndarray of shape (n_groups,)
        Fairness metric (phi) estimate for each group after fitting.
    phi_by_group_ci : ndarray of shape (n_groups, 3)
        Confidence intervals for each group after fitting (center, lower bound, upper bound).
    std : float
        Weighted standard deviation of phi values across groups after fitting.
    weights : ndarray of shape (n_groups,)
        Number of samples in each group after fitting.
    """
    def fit(self, s, y):
        """
        Fit the FairGroups partitioning to the data.

        Parameters
        ----------
        s : array-like
            Sensitive attribute values.
        y : array-like
            Binary outcome variable.

        Returns
        -------
        self : object
            Fitted instance with partition and statistics.
        """
        std = -np.inf
        partition = None
        weights = None
        phi_by_group = None

        s_grid, _, p_s1 = compute_phi_on_grid(s, y, self.grid_size)
        phi = p_s1 - np.mean(y)
        phi_dict = {
            (s_grid[i], s_grid[j]): phi[i, j]
            for i in range(len(s_grid) - 1)
            for j in range(i + 1, len(s_grid))
        }
        s_min = np.min(s)
        s_max = np.max(s)
        internal_points = s_grid[1:-1]
        s_sorted = np.sort(s)
        num_combinations = sum(
            1 for _ in itertools.combinations(internal_points, self.n_groups - 1)
        )
        for internal_split in tqdm(
            itertools.combinations(internal_points, self.n_groups - 1), total=num_combinations
        ):
            # Construct partition: [s_min, s_1, ..., s_{n_groups-1}, s_max]
            s_values = [s_min] + list(internal_split) + [s_max]
            # Compute counts in each interval [s_i, s_{i+1}]
            left_idxs = np.searchsorted(s_sorted, s_values[:-1], side="left")
            right_idxs = np.searchsorted(s_sorted, s_values[1:], side="right")
            weights_tmp = right_idxs - left_idxs
            phi_by_group_tmp = np.array(
                [phi_dict[s_values[i], s_values[i + 1]] for i in range(self.n_groups)]
            )
            std_tmp = _compute_std(weights_tmp, phi_by_group_tmp)
            if std_tmp > std:
                std = std_tmp
                partition = list(s_values)
                weights = weights_tmp
                phi_by_group = phi_by_group_tmp
        self.std = std
        self.partition = partition
        self.weights = weights
        self.phi_by_group = phi_by_group
        groups = self.predict(s)
        phi_by_group_ci = _compute_phi_by_group_ci(groups, y)
        self.phi_by_group_ci = phi_by_group_ci

        return self


class FairKMeans(FairPartitionBase):
    """
    Partition continuous sensitive attribute into groups using k-means clustering.
    
    This implementation uses a k-means-like approach to partition the sensitive attribute
    into groups that maximize the variance of fairness criterion phi. Instead of
    clustering the raw sensitive attribute values, it clusters the phi values computed
    on adjacent grid intervals, then maps the cluster assignments back to partition
    boundaries.
    
    The algorithm works by:
    1. Creating a grid and computing phi values for adjacent intervals
    2. Using k-means to cluster these phi values into n_groups clusters
    3. Mapping cluster assignments back to grid points to create partition boundaries
    4. Computing final statistics on the resulting partition
    
    Parameters
    ----------
    n_groups : int
        Number of groups to partition the sensitive attribute into.
    grid_size : int, default=100
        Number of grid points for discretization and phi computation.
        
    Attributes
    ----------
    partition : list
        Partition boundaries defining the groups after fitting.
    phi_by_group : ndarray of shape (n_groups,)
        Fairness metric (phi) estimate for each group after fitting.
    phi_by_group_ci : ndarray of shape (n_groups, 3)
        Confidence intervals for each group after fitting (center, lower bound, upper bound).
    std : float
        Weighted standard deviation of phi values across groups after fitting.
    weights : ndarray of shape (n_groups,)
        Number of samples in each group after fitting.
        
    Notes
    -----
    This method only works in the case of monotonic fairness with respect to the sensitive attribute. 
    See the paper for more details.
    """
    def fit(self, s, y):
        """
        Fit the FairKMeans partitioning to the data.

        Parameters
        ----------
        s : array-like
            Sensitive attribute values.
        y : array-like
            Binary outcome variable.

        Returns
        -------
        self : object
            Fitted instance with partition and statistics.
        """
        weights = np.zeros(self.n_groups)
        phi_by_group = np.zeros(self.n_groups)
        s_grid, _, p_s1 = compute_phi_on_grid(s, y, self.grid_size)
        phi_matrix = p_s1 - np.mean(y)
        # Take Phi values on groups [s_0, s_1], [s_1, s_2], ... , [s_{n_groups-1}, s_{n_groups}]
        phi = np.diag(phi_matrix, k=1)
        km = KMeans(n_clusters=self.n_groups)
        km.fit(phi.reshape(-1, 1))
        
        partition = []
        for i in range(self.n_groups):
            partition.append(s_grid[np.where(km.labels_ == i)[0][0]])
        partition.append(np.max(s_grid))
        partition.sort()

        partition_idxs = np.searchsorted(s_grid, partition, side="left")
        for i in range(self.n_groups):
            phi_by_group[i] = phi_matrix[partition_idxs[i]][partition_idxs[i + 1]]
            weights[i] = np.sum((s >= partition[i]) & (s <= partition[i + 1]))
        
        self.partition = partition
        self.weights = weights
        self.phi_by_group = phi_by_group

        groups = self.predict(s)
        phi_by_group_ci = _compute_phi_by_group_ci(groups, y)
        self.phi_by_group_ci = phi_by_group_ci

        std = _compute_std(weights, phi_by_group)
        self.std = std

        return self
    