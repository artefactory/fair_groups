import numpy as np
import itertools
from tqdm import tqdm
from sklearn.cluster import KMeans
from .fairness_metrics import compute_phi_on_grid


def _compute_weights(parition, s, y):
    pass


def _compute_phi_by_group(partition, s, y):
    pass


def _compute_variance(weights, phi_by_group):
    sum_w = np.sum(weights)
    mean = np.sum(weights * phi_by_group) / sum_w
    variance = (phi_by_group - mean)**2
    variance = np.sum(weights * variance) / sum_w
    variance = np.sqrt(max(variance, 0))
    
    return variance


def compute_fairseg_partition(nb_groups, s, y, nb_points=100):
    variance = -np.inf
    partition = None
    weights = None
    phi_by_group = None

    s_grid, _, p_s1 = compute_phi_on_grid(s, y, nb_points)

    phi = p_s1 - np.mean(y)
    phi_dict = {(s_grid[i], s_grid[j]): phi[i, j] 
                for i in range(len(s_grid)-1) 
                for j in range(i+1, len(s_grid))}

    s_min = np.min(s)
    s_max = np.max(s)
    internal_points = s_grid[1:-1]
    s_sorted = np.sort(s)

    num_combinations = sum(1 for _ in itertools.combinations(internal_points, nb_groups - 1))

    # Select (nb_groups - 1) internal split points to form nb_groups intervals
    for internal_split in tqdm(itertools.combinations(internal_points, nb_groups - 1), total=num_combinations):
        # Construct partition: [s_min, s_1, ..., s_{nb_groups-1}, s_max]
        s_values = [s_min] + list(internal_split) + [s_max]

        # Compute counts in each interval [s_i, s_{i+1}]
        left_idxs = np.searchsorted(s_sorted, s_values[:-1], side='left')
        right_idxs = np.searchsorted(s_sorted, s_values[1:], side='right')
        weights_tmp = right_idxs - left_idxs

        phi_by_group_tmp = np.array([phi_dict[s_values[i], s_values[i+1]] for i in range(nb_groups)])

        var_tmp = _compute_variance(weights_tmp, phi_by_group_tmp)

        if var_tmp > variance:
            variance = var_tmp
            partition = list(s_values)
            weights = weights_tmp
            phi_by_group = phi_by_group_tmp
    
    result = {'var' : variance, 'partition' : partition, 'weights' : weights, 'phi' : phi_by_group}
    return result


def compute_k_means_partition(nb_groups, s, y, nb_points=100):
    weights = np.zeros(nb_groups)
    phi_by_group = np.zeros(nb_groups)
    
    s_grid, _, p_s1 = compute_phi_on_grid(s, y, nb_points)
    phi_matrix = p_s1 - np.mean(y)
    # Take Phi values on groups [s_0, s_1], [s_1, s_2], ... , [s_{nb_groups-1}, s_{nb_groups}]
    phi = np.diag(phi_matrix, k=1)
    
    km = KMeans(n_clusters=nb_groups)
    km.fit(phi.reshape(-1,1))
    
    partition = []
    for i in range(nb_groups):
        partition.append(s_grid[np.where(km.labels_ == i)[0][0]])
    partition.append(np.max(s_grid))
    partition.sort()
    
    partition_idxs = np.searchsorted(s_grid, partition, side='left')
    for i in range(nb_groups):
        phi_by_group[i] = phi_matrix[partition_idxs[i]][partition_idxs[i+1]]
        weights[i] = np.sum((s >= partition[i]) & (s <= partition[i+1]))
     
    variance = _compute_variance(weights, phi_by_group)
    
    result = {'var' : variance, 'partition' : partition, 'weights' : weights, 'phi' : phi_by_group}
    return result


def compute_partition(method : str, nb_groups, s, y, nb_points=100):
    method_map = {
        "fairseg": compute_fairseg_partition,
        "k-means": compute_k_means_partition,
    }

    if method not in method_map:
        available_methods = ', '.join(method_map.keys())
        raise ValueError(f"Unknown partition method: {method}. Available methods are: {available_methods}")

    return method_map[method](nb_groups, s, y, nb_points)


def compute_partition_metrics(partition, s, y):
    weights = _compute_weights(partition, s, y)
    phi_by_group = _compute_phi_by_group(partition, s, y)
    variance = _compute_variance(weights, phi_by_group)
    
    result = {'var' : variance, 'partition' : partition, 'weights' : weights, 'phi' : phi_by_group}
    return result
