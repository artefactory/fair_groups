# Author: Jean-Michel Loubes <jm.loubes@gmail.com>
# Laurent Risser <laurent.risser@math.univ-toulouse.fr>
# Veronika Shilova <veronika.shilova@artefact.com>


import numpy as np
import scipy


def di_sliding_window(Y, S, window_size=5, alpha=.95, grid_size=100):
    """
    Calculate disparate impact (DI) and its corresponding confidence interval 
    for continuous sensitive variable with sliding window approach.
    
    Parameters
    ----------
    Y : numpy array
        Ground truth or predicted class - {0, 1}.
    S : numpy array
        Continuous sensitive variable.
    window_size : float
        Defines the discriminated group - [s-window_size, s+window_size].
    alpha : float
        Quartile for the desired confidence interval.
    grid_size : int
        Number of points in discretization of S.

    Returns
    -------
    numpy array
        DI and its corresponding confidence interval.
    """
    S_grid = np.linspace(np.min(S)+window_size, np.max(S)-window_size, grid_size)
    di = [_compute_di(Y, ((S < s-window_size) | (S > s+window_size)), alpha=alpha) for s in S_grid]
    
    return np.array(di), S_grid

def di_threshold(Y, S, alpha=.95, grid_size=100):
    """
    Calculate disparate impact (DI) and its corresponding confidence interval 
    for continuous sensitive variable with threshold approach.
    
    Parameters
    ----------
    Y : numpy array
        Ground truth or predicted class - {0, 1}.
    S : numpy array
        Continuous sensitive variable.
    alpha : float
        Quartile for the desired confidence interval.
    grid_size : int
        Number of points in discretization of S.

    Returns
    -------
    numpy array
        DI and its corresponding confidence interval.
    """
    S_grid = np.linspace(np.min(S), np.max(S), grid_size, endpoint=False)
    di = [_compute_di(Y, (S > s), alpha=alpha) for s in S_grid]
    
    return np.array(di), S_grid

def tpr_sliding_window(Y, Y_pred, S, window_size=5, alpha=.95, grid_size=100):
    """
    Calculate True Positive Rate (TPR) or Recall and its corresponding confidence interval 
    for continuous sensitive variable with sliding window approach.
    
    Parameters
    ----------
    Y : numpy array
        Ground truth class - {0, 1}.
    Y_pred : numpy array
        Predicted class - {0, 1}.
    S : numpy array
        Continuous sensitive variable.
    window_size : float
        Defines the discriminated group - [s-window_size, s+window_size].
    alpha : float
        Quartile for the desired confidence interval.
    grid_size : int
        Number of points in discretization of S.

    Returns
    -------
    numpy array
        TPR and its corresponding confidence interval.
    """
    S_grid = np.linspace(np.min(S)+window_size, np.max(S)-window_size, grid_size)
    tpr = [_compute_eo(Y, Y_pred, ((S < s-window_size) | (S > s+window_size)), alpha=alpha) for s in S_grid]
    
    return np.array(tpr), S_grid

def tpr_threshold(Y, Y_pred, S, alpha=.95, grid_size=100):
    """
    Calculate True Positive Rate (TPR) or Recall and its corresponding confidence interval 
    for continuous sensitive variable with threshold approach.
    
    Parameters
    ----------
    Y : numpy array
        Ground truth class - {0, 1}.
    Y_pred : numpy array
        Predicted class - {0, 1}.
    S : numpy array
        Continuous sensitive variable.
    alpha : float
        Quartile for the desired confidence interval.
    grid_size : int
        Number of points in discretization of S.

    Returns
    -------
    numpy array
        TPR and its corresponding confidence interval.
    """
    S_grid = np.linspace(np.min(S), np.max(S), grid_size, endpoint=False)
    tpr = [_compute_eo(Y, Y_pred, (S > s), alpha=alpha) for s in S_grid]
    
    return np.array(tpr), S_grid

def tnr_sliding_window(Y, Y_pred, S, window_size=5, alpha=.95, grid_size=100):
    """
    Calculate True Negative Rate (TNR) and its corresponding confidence interval 
    for continuous sensitive variable with sliding window approach.
    
    Parameters
    ----------
    Y : numpy array
        Ground truth class - {0, 1}.
    Y_pred : numpy array
        Predicted class - {0, 1}.
    S : numpy array
        Continuous sensitive variable.
    window_size : float
        Defines the discriminated group - [s-window_size, s+window_size].
    alpha : float
        Quartile for the desired confidence interval.
    grid_size : int
        Number of points in discretization of S.

    Returns
    -------
    numpy array
        TNR and its corresponding confidence interval.
    """
    S_grid = np.linspace(np.min(S)+window_size, np.max(S)-window_size, grid_size)
    tnr = [_compute_eo(1-Y, 1-Y_pred, ((S < s-window_size) | (S > s+window_size)), alpha=alpha) for s in S_grid]
    
    return np.array(tnr), S_grid

def tnr_threshold(Y, Y_pred, S, alpha=.95, grid_size=100):
    """
    Calculate True Negative Rate (TNR) and its corresponding confidence interval 
    for continuous sensitive variable with threshold approach.
    
    Parameters
    ----------
    Y : numpy array
        Ground truth class - {0, 1}.
    Y_pred : numpy array
        Predicted class - {0, 1}.
    S : numpy array
        Continuous sensitive variable.
    alpha : float
        Quartile for the desired confidence interval.
    grid_size : int
        Number of points in discretization of S.

    Returns
    -------
    numpy array
        TNR and its corresponding confidence interval.
    """
    S_grid = np.linspace(np.min(S), np.max(S), grid_size, endpoint=False)
    tnr = [_compute_eo(1-Y, 1-Y_pred, (S > s), alpha=alpha) for s in S_grid]
    
    return np.array(tnr), S_grid

def sufficiency_sliding_window(Y, Y_pred, S, window_size=5, alpha=.95, grid_size=100):
    """
    Calculate Sufficiency and its corresponding confidence interval 
    for continuous sensitive variable with sliding window approach.
    
    Parameters
    ----------
    Y : numpy array
        Ground truth class - {0, 1}.
    Y_pred : numpy array
        Predicted class - {0, 1}.
    S : numpy array
        Continuous sensitive variable.
    window_size : float
        Defines the discriminated group - [s-window_size, s+window_size].
    alpha : float
        Quartile for the desired confidence interval.
    grid_size : int
        Number of points in discretization of S.

    Returns
    -------
    numpy array
        Sufficiency and its corresponding confidence interval.
    """
    S_grid = np.linspace(np.min(S)+window_size, np.max(S)-window_size, grid_size)
    sufficiency = [_compute_sufficiency(Y, Y_pred, ((S < s-window_size) | (S > s+window_size)), alpha=alpha) for s in S_grid]
    
    return np.array(sufficiency), S_grid

def sufficiency_threshold(Y, Y_pred, S, alpha=.95, grid_size=100):
    """
    Calculate Sufficiency and its corresponding confidence interval 
    for continuous sensitive variable with threshold approach.
    
    Parameters
    ----------
    Y : numpy array
        Ground truth class - {0, 1}.
    Y_pred : numpy array
        Predicted class - {0, 1}.
    S : numpy array
        Continuous sensitive variable.
    alpha : float
        Quartile for the desired confidence interval.
    grid_size : int
        Number of points in discretization of S.

    Returns
    -------
    numpy array
        Sufficiency and its corresponding confidence interval.
    """
    S_grid = np.linspace(np.min(S), np.max(S), grid_size, endpoint=False)
    sufficiency = [_compute_sufficiency(Y, Y_pred, (S > s), alpha=alpha) for s in S_grid]
    
    return np.array(sufficiency), S_grid

def subgroup_fairness():
    pass

def _compute_di(Y, S, alpha):
    Z = np.array([np.multiply(1-S, Y), np.multiply(S, Y), 1-S, S])
    return _asymptotic_behavior(Z, alpha)

def _compute_eo(Y, Y_pred, S, alpha):
    Z = np.array([np.multiply(1-S, Y_pred), np.multiply(S, Y_pred), 1-S, S])
    Z = np.multiply(Z, Y)
    return _asymptotic_behavior(Z, alpha)

def _compute_sufficiency(Y, Y_pred, S, alpha):
    Z = np.array([np.multiply(1-S, Y), np.multiply(S, Y), 1-S, S])
    Z = np.multiply(Z, Y_pred)
    return _asymptotic_behavior(Z, alpha)

def _asymptotic_behavior(Z, alpha):
    """
    Define asymptotic behavior of DI/TPR/TNR/Sufficiency estimators using the delta method.
    For more information please refer to the paper: Besse, Philippe, et al. 
    "Confidence intervals for testing disparate impact in fair learning." 
    arXiv preprint arXiv:1807.06362 (2018).
    """
    # Number of observations
    n = Z.shape[1]
    
    # Expected value and covariance matrix 
    E = np.mean(Z, axis=1)
    cov_matrix = np.cov(Z, ddof=0)

    #phi gradient applied to E
    grad_phi_E_T = [E[3]/(E[1]*E[2]), -(E[0]*E[3])/(E[1]**2*E[2]), -(E[0]*E[3])/(E[1]*E[2]**2), E[0]/(E[1]*E[2])]
    grad_phi_E_T = np.nan_to_num(grad_phi_E_T)

    #confidence interval
    center, ic_inf, ic_sup = _create_IC(n, E, cov_matrix, grad_phi_E_T, alpha)
    
    return [center, ic_inf, ic_sup]

def _create_IC(nb_obs, E, cov_matrix, grad_phi_E_T, alpha):
    # Construct confidence interval for DI/TPR/TNR/Sufficiency estimators
    center = (E[0] * E[3]) / (E[1] * E[2]) if (E[1] * E[2]) else .0
    sigma = np.dot(grad_phi_E_T, np.dot(cov_matrix, np.transpose(grad_phi_E_T)))
    norm_quartile = scipy.stats.norm.ppf((1 - alpha)*.5)
    inter = (sigma / nb_obs)**.5
    radius = inter * norm_quartile
    
    return center, center-radius, center+radius