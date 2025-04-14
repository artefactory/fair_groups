import numpy as np
import itertools
from tqdm import tqdm

from .fairness_metrics import phi_on_grid


class FairSeg:
    def __init__(self, k):
        self.k = k 
        
        self.variance = -np.inf
        self.partition = None
        self.weights = None
        self.phi_by_group = None
        
        
    def compute_partition(self, s, y, nb_points=100):
        s_grid, _, p_s1 = phi_on_grid(s, y, nb_points)

        phi = p_s1 - np.mean(y)
        phi_dict = {(s_grid[i], s_grid[j]): phi[i, j] 
                    for i in range(len(s_grid)-1) 
                    for j in range(i+1, len(s_grid))}

        s_min = np.min(s)
        s_max = np.max(s)
        internal_points = s_grid[1:-1]
        s_sorted = np.sort(s)

        num_combinations = sum(1 for _ in itertools.combinations(internal_points, self.k - 1))

        # Select (k - 1) internal split points to form k intervals
        for internal_split in tqdm(itertools.combinations(internal_points, self.k - 1), total=num_combinations):
            # Construct partition: [s_min, s_1, ..., s_{k-1}, s_max]
            s_values = [s_min] + list(internal_split) + [s_max]

            # Compute counts in each interval [l_i, l_{i+1}]
            left_idxs = np.searchsorted(s_sorted, s_values[:-1], side='left')
            right_idxs = np.searchsorted(s_sorted, s_values[1:], side='right')
            weights = right_idxs - left_idxs

            phi_by_group = np.array([phi_dict[s_values[i], s_values[i+1]] for i in range(self.k)])

            sum_w = np.sum(weights)
            mean = np.sum(weights * phi_by_group) / sum_w
            var_tmp = (phi_by_group - mean)**2
            var_tmp = np.sum(weights * var_tmp) / sum_w
            var_tmp = np.sqrt(max(var_tmp, 0))

            if var_tmp > self.variance:
                self.variance = var_tmp
                self.partition = list(s_values)
                self.weights = weights.copy()
                self.phi_by_group = phi_by_group.copy()
                