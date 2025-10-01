import numpy as np
import scipy

# Threshold for Phi estimation to avoid zero division
zero_thr = 1e-12


def compute_phi_sp_ci(s, y, alpha=0.95):
    """
    Calculate confidence interval of the estimator of [P(Y = 1 | S = 1) - P(Y = 1)] using the delta method.

    For more information, see:
    Besse, Philippe, et al. "Confidence intervals for testing disparate impact in fair learning."
    arXiv preprint arXiv:1807.06362 (2018).

    Parameters
    ----------
    s : array-like, type=boolean
        Binary sensitive attribute (0/1 or boolean mask).
    y : array-like
        Binary outcome variable.
    alpha : float, default=0.95
        Confidence level.

    Returns
    -------
    ci : list of float
        [center, lower_bound, upper_bound] of the confidence interval.
    """
    Z = np.array([np.multiply(s, y), s, y, 1 - y])
    n = Z.shape[1]

    # Expected value and covariance matrix
    E = np.mean(Z, axis=1)
    cov_matrix = np.cov(Z, ddof=0)

    # phi gradient applied to E
    grad_phi_E_T = [
        1.0 / E[1],
        -E[0] / E[1] ** 2,
        E[3] / (E[2] + E[3]) ** 2,
        -E[2] / (E[2] + E[3]) ** 2,
    ]

    # confidence interval
    center = E[0] / E[1] - (E[2] / (E[2] + E[3]))
    sigma = np.dot(grad_phi_E_T, np.dot(cov_matrix, np.transpose(grad_phi_E_T)))
    norm_quartile = scipy.stats.norm.ppf((1 + alpha) * 0.5)
    inter = (sigma / n) ** 0.5
    radius = inter * norm_quartile

    return [center, center - radius, center + radius]


def compute_phi_on_grid(
    s, y, grid_size=100, bootstrap_ratio=1.0, n_points_for_weights=2000, weights=None
):
    """
    Compute the conditional probability P(Y=1|S) estimate on a grid of sensitive attribute values.

    Parameters
    ----------
    s : array-like type=float
        Sensitive attribute values, expressed as continous quantities.
    y : array-like, type=boolean
        Binary outcome variable.
    grid_size : int, default=100
        Number of grid points for partitioning.
    bootstrap_ratio : float, default=1.0
        Ratio of samples to use for bootstrapping.
    n_points_for_weights : int, default=2000
        Number of bins for rebalancing weights.
    weights : array-like, optional
        Optional weights for rebalancing.

    Returns
    -------
    s_grid : ndarray
        Grid of sensitive attribute values.
    matrix_p_s0 : ndarray
        Conditional probabilities for S=0.
    matrix_p_s1 : ndarray
        Conditional probabilities for S=1.
    """
    s_grid_min = np.min(s)
    s_grid_max = np.max(s)
    s_grid = np.linspace(s_grid_min, s_grid_max, grid_size)
    s_bins = np.linspace(s_grid_min, s_grid_max, n_points_for_weights)

    def iteratively_count(n_by_s_grid):
        counter_s0 = np.zeros((grid_size, grid_size))
        counter_s1 = np.zeros((grid_size, grid_size))
        counter_s0[0, 1] = np.sum(n_by_s_grid[1:])
        counter_s1[0, 1] = n_by_s_grid[0]
        for j in range(2, len(s_grid)):
            counter_s0[0, j] = counter_s0[0, j - 1] - n_by_s_grid[j - 1]
            counter_s1[0, j] = counter_s1[0, j - 1] + n_by_s_grid[j - 1]

        for i in range(1, len(s_grid)):
            for j in range(i + 1, len(s_grid)):
                counter_s0[i, j] = counter_s0[i - 1, j] + n_by_s_grid[i - 1]
                counter_s1[i, j] = counter_s1[i - 1, j] - n_by_s_grid[i - 1]

        return counter_s1, counter_s0

    bootstrapped = np.random.choice(range(len(s)), int(len(s) * bootstrap_ratio), replace=False)
    s_bootstrapped = s[bootstrapped]
    y_bootstrapped = y[bootstrapped]

    rebalance_weights = np.zeros(len(y_bootstrapped))
    if weights:
        n_s_by_bins = [
            np.sum((weights > s_bins[i - 1]) & (weights <= s_bins[i]))
            for i in range(1, len(s_bins))
        ]
        for i in range(1, len(s_bins)):
            rebalance_weights[(s_bootstrapped > s_bins[i - 1]) * (s_bootstrapped <= s_bins[i])] = (
                n_s_by_bins[i - 1]
            )
    else:
        rebalance_weights = np.ones(len(y_bootstrapped))
    rebalance_weights /= rebalance_weights.sum()

    n_samples = []
    n_pos_y = []
    for i in range(1, len(s_grid)):
        mask = (s_bootstrapped > s_grid[i - 1]) & (s_bootstrapped <= s_grid[i])
        n_samples.append(np.sum(rebalance_weights[mask]))
        n_pos_y.append(np.sum(rebalance_weights[mask] * y_bootstrapped[mask]))

    matrix_n_pos_y_s1, matrix_n_pos_y_s0 = iteratively_count(n_pos_y)
    matrix_n_samples_s1, matrix_n_samples_s0 = iteratively_count(n_samples)
    matrix_p_s1 = matrix_n_samples_s1.copy()
    matrix_p_s0 = matrix_n_samples_s0.copy()
    matrix_p_s1[matrix_p_s1 > zero_thr] = (
        matrix_n_pos_y_s1[matrix_p_s1 > zero_thr] / matrix_n_samples_s1[matrix_p_s1 > zero_thr]
    )
    matrix_p_s0[matrix_p_s0 > zero_thr] = (
        matrix_n_pos_y_s0[matrix_p_s0 > zero_thr] / matrix_n_samples_s0[matrix_p_s0 > zero_thr]
    )

    return s_grid, matrix_p_s0, matrix_p_s1


def get_conditional_positive_y_proba(s, y, n_bins=100):
    """
    Calculates the grid of s_bins and the conditional probability P(Y=1|S) for each bin.

    Parameters
    ----------
    s : array-like, type=float
        Sensitive attribute values, expressed as continous quantities.
    y : array-like, type=boolean
        Binary outcome variable.
    n_bins : int, default=100
        Number of bins for s.

    Returns
    -------
    s_bins : ndarray
        Grid of sensitive attribute values.
    y_pred_proba : ndarray
        Probability of Y=1 for each bin.
    """
    s_min = np.min(s)
    s_max = np.max(s)
    s_bins = np.linspace(s_min, s_max, n_bins)
    y_pred_proba = [
        np.mean(y[(s >= s_bins[i]) & (s <= s_bins[i + 1])]) for i in range(len(s_bins) - 1)
    ]
    return s_bins, np.array(y_pred_proba)
