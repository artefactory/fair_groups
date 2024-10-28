import numpy as np
from .utils import _compute_di, _compute_eo, _compute_sufficiency


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
    S_grid = np.linspace(np.min(S), np.max(S), grid_size)
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