import numpy as np
import itertools
from tqdm import tqdm
from sklearn.cluster import KMeans
from .fairness_metrics import compute_phi_on_grid, compute_phi_sp_ci


def _compute_weights(partition, s):
    s_sorted = np.sort(s)
    # Compute counts in each interval [partition[i], partition[i+1]]
    left_idxs = np.searchsorted(s_sorted, partition[:-1], side='left')
    right_idxs = np.searchsorted(s_sorted, partition[1:], side='right')
    weights = right_idxs - left_idxs
    
    return weights


def _compute_phi_by_group(partition, s, y):
    phi_by_group = np.full(len(partition)-1, np.nan)

    for i in range(self.n_groups):
        # Select values where s is in [partition[i], partition[i+1]]
        group = (s >= partition[i]) & (s <= partition[i+1])
        if np.any(group):
            phi_by_group[group] = y[group].mean()
  
    return phi_by_group


def _compute_phi_by_group_ci(partition, s, y):
    phi_by_group_ci = [compute_phi_sp_ci((s >= partition[i]) & (s <= partition[i+1]), y) for i in range(len(partition)-1)]
    return np.array(phi_by_group_ci)


def _compute_variance(weights, phi_by_group):
    sum_w = np.sum(weights)
    mean = np.sum(weights * phi_by_group) / sum_w
    variance = (phi_by_group - mean)**2
    variance = np.sum(weights * variance) / sum_w
    variance = np.sqrt(max(variance, 0))
    
    return variance


class FairGroups:
    def __init__(self, nb_groups, nb_points=100):
        self.nb_groups = nb_groups
        self.nb_points = nb_points
        
    def fit(self, s, y):
        variance = -np.inf
        partition = None
        weights = None
        phi_by_group = None

        s_grid, _, p_s1 = compute_phi_on_grid(s, y, self.nb_points)

        phi = p_s1 - np.mean(y)
        phi_dict = {(s_grid[i], s_grid[j]): phi[i, j] 
                    for i in range(len(s_grid)-1) 
                    for j in range(i+1, len(s_grid))}

        s_min = np.min(s)
        s_max = np.max(s)
        internal_points = s_grid[1:-1]
        s_sorted = np.sort(s)

        num_combinations = sum(1 for _ in itertools.combinations(internal_points, self.nb_groups - 1))

        # Select (nb_groups - 1) internal split points to form nb_groups intervals
        for internal_split in tqdm(itertools.combinations(internal_points, self.nb_groups - 1), total=num_combinations):
            # Construct partition: [s_min, s_1, ..., s_{nb_groups-1}, s_max]
            s_values = [s_min] + list(internal_split) + [s_max]

            # Compute counts in each interval [s_i, s_{i+1}]
            left_idxs = np.searchsorted(s_sorted, s_values[:-1], side='left')
            right_idxs = np.searchsorted(s_sorted, s_values[1:], side='right')
            weights_tmp = right_idxs - left_idxs

            phi_by_group_tmp = np.array([phi_dict[s_values[i], s_values[i+1]] for i in range(self.nb_groups)])

            var_tmp = _compute_variance(weights_tmp, phi_by_group_tmp)

            if var_tmp > variance:
                variance = var_tmp
                partition = list(s_values)
                weights = weights_tmp
                phi_by_group = phi_by_group_tmp

        self.variance = variance
        self.partition = partition
        self.weights = weights
        self.phi_by_group = phi_by_group
        
        phi_by_group_ci = _compute_phi_by_group_ci(partition, s, y)
        self.phi_by_group_ci = phi_by_group_ci
        
        return self
    
    def predict(self, s, y):
        phi_by_group = _compute_phi_by_group(self.partition, s, y)
        phi_by_group_ci = _compute_phi_by_group_ci(self.partition, s, y)
        weights = _compute_weights(self.partition, s)
        variance = _compute_variance(weights, phi_by_group)
        
        return phi_by_group, phi_by_group_ci, variance


class FairKMeans:
    def __init__(self, nb_groups, nb_points=100):
        self.nb_groups = nb_groups
        self.nb_points = nb_points
    
    def fit(self, s, y):
        weights = np.zeros(self.nb_groups)
        phi_by_group = np.zeros(self.nb_groups)

        s_grid, _, p_s1 = compute_phi_on_grid(s, y, self.nb_points)
        phi_matrix = p_s1 - np.mean(y)
        # Take Phi values on groups [s_0, s_1], [s_1, s_2], ... , [s_{nb_groups-1}, s_{nb_groups}]
        phi = np.diag(phi_matrix, k=1)

        km = KMeans(n_clusters=self.nb_groups)
        km.fit(phi.reshape(-1,1))

        partition = []
        for i in range(self.nb_groups):
            partition.append(s_grid[np.where(km.labels_ == i)[0][0]])
        partition.append(np.max(s_grid))
        partition.sort()

        partition_idxs = np.searchsorted(s_grid, partition, side='left')
        for i in range(self.nb_groups):
            phi_by_group[i] = phi_matrix[partition_idxs[i]][partition_idxs[i+1]]
            weights[i] = np.sum((s >= partition[i]) & (s <= partition[i+1]))

        variance = _compute_variance(weights, phi_by_group)
        phi_by_group_ci = _compute_phi_by_group_ci(partition, s, y)

        self.variance = variance
        self.partition = partition
        self.weights = weights
        self.phi_by_group = phi_by_group
        self.phi_by_group_ci = phi_by_group_ci
        
        return self
    
    def predict(self, s, y):
        phi_by_group = _compute_phi_by_group(self.partition, s, y)
        phi_by_group_ci = _compute_phi_by_group_ci(self.partition, s, y)
        weights = _compute_weights(self.partition, s)
        variance = _compute_variance(weights, phi_by_group)
        
        return phi_by_group, phi_by_group_ci, variance
