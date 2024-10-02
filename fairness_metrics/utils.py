# Author: Jean-Michel Loubes <jm.loubes@gmail.com>
# Laurent Risser <laurent.risser@math.univ-toulouse.fr>
# Veronika Shilova <veronika.shilova@artefact.com>

import numpy as np
import scipy


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
    inter = (sigma / nb_obs) ** .5
    radius = inter * norm_quartile
    
    return center, center - radius, center + radius
