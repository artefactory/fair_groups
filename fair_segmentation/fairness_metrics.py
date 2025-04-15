import numpy as np
import scipy


# Threshold for Phi estimation to avoid zero division
zero_thr = 1e-12


def compute_phi_sp_ci(s, y, alpha=.95):
    """
    Calculate confidence interval of the estimator of [P(Y = 1 | S = 1) - P(Y = 1)] using the delta method.
    For more information please refer to the paper: Besse, Philippe, et al. 
    "Confidence intervals for testing disparate impact in fair learning." 
    arXiv preprint arXiv:1807.06362 (2018).
    """
    Z = np.array([np.multiply(s, y), s, y, 1-y])
    n = Z.shape[1]
    
    # Expected value and covariance matrix 
    E = np.mean(Z, axis=1)
    cov_matrix = np.cov(Z, ddof=0)

    #phi gradient applied to E
    grad_phi_E_T = [1. / E[1], -E[0] / E[1]**2, E[3] / (E[2] + E[3])**2, -E[2] / (E[2] + E[3])**2]

    #confidence interval
    center = E[0] / E[1] - (E[2] / (E[2] + E[3]))
    sigma = np.dot(grad_phi_E_T, np.dot(cov_matrix, np.transpose(grad_phi_E_T)))
    norm_quartile = scipy.stats.norm.ppf((1 + alpha)*.5)
    inter = (sigma / n) ** .5
    radius = inter * norm_quartile
    
    return [center, center - radius, center + radius]
    

def compute_phi_on_grid(s, y, nb_points=100, bootstrap_ratio=1., nb_points_for_weights=2000, weights=None):
    s_grid_min = np.min(s)
    s_grid_max = np.max(s)
    s_grid = np.linspace(s_grid_min, s_grid_max, nb_points)
    s_bins = np.linspace(s_grid_min, s_grid_max, nb_points_for_weights)

    def iteratively_count(nb_by_s_grid):
        counter_s0 = np.zeros((nb_points, nb_points))
        counter_s1 = np.zeros((nb_points, nb_points))

        # Initialize first row
        counter_s0[0, 1] = np.sum(nb_by_s_grid[1:])
        counter_s1[0, 1] = nb_by_s_grid[0]
        for j in range(2, len(s_grid)):
            counter_s0[0, j] = counter_s0[0, j-1] - nb_by_s_grid[j-1]
            counter_s1[0, j] = counter_s1[0, j-1] + nb_by_s_grid[j-1]
            
        for i in range(1, len(s_grid)):
            for j in range(i+1, len(s_grid)):
                counter_s0[i, j] = counter_s0[i-1, j] + nb_by_s_grid[i-1]
                counter_s1[i, j] = counter_s1[i-1, j] - nb_by_s_grid[i-1]

        return counter_s1, counter_s0

    bootstrapped = np.random.choice(range(len(s)), int(len(s) * bootstrap_ratio), replace=False)
    s_bootstrapped = s[bootstrapped]
    y_bootstrapped = y[bootstrapped]

    rebalance_weights = np.zeros(len(y_bootstrapped))
    if weights:
        nb_s_by_bins = [np.sum((weights > s_bins[i-1]) & (weights <= s_bins[i])) for i in range(1, len(s_bins))]
        for i in range(1, len(s_bins)):
            rebalance_weights[(s_bootstrapped > s_bins[i-1]) * (s_bootstrapped <= s_bins[i])] = nb_s_by_bins[i - 1]
    else:
        rebalance_weights = np.ones(len(y_bootstrapped)) 
    rebalance_weights /= rebalance_weights.sum()

    nb_samples = []
    nb_pos_y = []
    for i in range(1, len(s_grid)):
        mask = (s_bootstrapped > s_grid[i-1]) & (s_bootstrapped <= s_grid[i])
        nb_samples.append(np.sum(rebalance_weights[mask]))
        nb_pos_y.append(np.sum(rebalance_weights[mask] * y_bootstrapped[mask]))

    matrix_n_pos_y_s1, matrix_n_pos_y_s0 = iteratively_count(nb_pos_y)
    matrix_n_samples_s1, matrix_n_samples_s0 = iteratively_count(nb_samples)
    matrix_p_s1 = matrix_n_samples_s1.copy()
    matrix_p_s0 = matrix_n_samples_s0.copy()
    matrix_p_s1[matrix_p_s1 > zero_thr] = matrix_n_pos_y_s1[matrix_p_s1 > zero_thr] / matrix_n_samples_s1[matrix_p_s1 > zero_thr]
    matrix_p_s0[matrix_p_s0 > zero_thr] = matrix_n_pos_y_s0[matrix_p_s0 > zero_thr] / matrix_n_samples_s0[matrix_p_s0 > zero_thr]

    return s_grid, matrix_p_s0, matrix_p_s1
