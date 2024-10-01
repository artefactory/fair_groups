import numpy as np
import matplotlib.pyplot as plt

def plot_di_sliding_window(di, di_pred, S_grid, window_size, 
                           sensitive_var_name, dataset_name, target_name, 
                           pred_target_name, di_color='plum', 
                           di_pred_color='royalblue'):
    """
    Plot disparate impact (DI) and its corresponding confidence interval
    for ground truth and predictions (calculated for continuous sensitive variable
    with sliding window approach).
    
    Parameters
    ----------
    di : numpy array
        DI calculated w.r.t. to ground truth.
    di_pred : numpy array
        DI calculated w.r.t. to predictions.
    S_grid : numpy array
        Discretization of continuous sensitive variable on which DI is calculated.
    window_size : float
        Defines the discriminated group - [s-window_size, s+window_size].
    sensitive_var_name : str
        Name of sensitive variable for x-axis.
    dataset_name : str
        Name of the dataset for plot title.
    target_name : str
        Name of target variable for legend.
    pred_target_name : str
        Name of predicted variable for legend.
    di_color : str
        Color for ground truth DI plot.
    di_pred_color : str
        Color for predictions DI plot.
    """
    plt.title(f'DI as function of range of ${sensitive_var_name}$ ($\pm$ {window_size}) ({dataset_name})')
    plt.plot(S_grid, di[:, 0], color=di_color, label=target_name)
    plt.fill_between(S_grid, di[:, 1], di[:, 2], color=di_color, alpha=.2)
    plt.plot(S_grid, di_pred[:, 0], color=di_pred_color, label=pred_target_name)
    plt.fill_between(S_grid, di_pred[:, 1], di_pred[:, 2], color=di_pred_color, alpha=.2)
    plt.xlabel(f'${sensitive_var_name}$')
    plt.ylabel('Disparate Impact (DI)')
    plt.legend()
    plt.show()

def plot_di_threshold(di, di_pred, S_grid, sensitive_var_name, 
                      dataset_name, target_name, pred_target_name, 
                      di_color='plum', di_pred_color='royalblue'):
    """
    Plot disparate impact (DI) and its corresponding confidence interval
    for ground truth and predictions (calculated for continuous sensitive variable
    with threshold approach).
    
    Parameters
    ----------
    di : numpy array
        DI calculated w.r.t. to ground truth.
    di_pred : numpy array
        DI calculated w.r.t. to predictions.
    S_grid : numpy array
        Discretization of continuous sensitive variable on which DI is calculated.
    sensitive_var_name : str
        Name of sensitive variable for x-axis.
    dataset_name : str
        Name of the dataset for plot title.
    target_name : str
        Name of target variable for legend.
    pred_target_name : str
        Name of predicted variable for legend.
    di_color : str
        Color for ground truth DI plot.
    di_pred_color : str
        Color for predictions DI plot.
    """
    plt.title(f'DI as function of threshold of ${sensitive_var_name}$ ({dataset_name})')
    plt.plot(S_grid, di[:, 0], color=di_color, label=target_name)
    plt.fill_between(S_grid, di[:, 1], di[:, 2], color=di_color, alpha=.2)
    plt.plot(S_grid, di_pred[:, 0], color=di_pred_color, label=pred_target_name)
    plt.fill_between(S_grid, di_pred[:, 1], di_pred[:, 2], color=di_pred_color, alpha=.2)
    plt.xlabel(f'${sensitive_var_name}$')
    plt.ylabel('Disparate Impact (DI)')
    plt.legend()
    plt.show()

def plot_eo_sliding_window(tpr, tnr, S_grid, window_size, 
                           sensitive_var_name, dataset_name, 
                           target_name, tpr_color='orange',
                           tnr_color='forestgreen'):
    """
    Plot Equality of Odds (EO) (True Positive Rate (TPR) or Recall and True Negative Rate (TNR))
    and its corresponding confidence interval for ground truth and predictions 
    (calculated for continuous sensitive variable with sliding window approach).
    
    Parameters
    ----------
    tpr : numpy array
        True Positive Rate (TPR) or Recall.
    tnr : numpy array
        True Negative Rate (TNR).
    S_grid : numpy array
        Discretization of continuous sensitive variable on which TPR, TNR are calculated.
    window_size : float
        Defines the discriminated group - [s-window_size, s+window_size].
    sensitive_var_name : str
        Name of sensitive variable for x-axis.
    dataset_name : str
        Name of the dataset for plot title.
    target_name : str
        Name of target variable for legend.
    tpr_color : str
        Color for TPR plot.
    tnr_pred_color : str
        Color for TNR plot.
    """
    plt.title(f'Equality of Odds as function of range of ${sensitive_var_name}$ ($\pm$ {window_size}) ({dataset_name})')
    plt.plot(S_grid, tpr[:, 0], color=tpr_color, label=f'TPR, {target_name}')
    plt.fill_between(S_grid, tpr[:, 1], tpr[:, 2], color=tpr_color, alpha=.2)
    plt.plot(S_grid, tnr[:, 0], color=tnr_color, label=f'TNR, {target_name}')
    plt.fill_between(S_grid, tnr[:, 1], tnr[:, 2], color=tnr_color, alpha=.2)
    plt.xlabel(f'${sensitive_var_name}$')
    plt.legend()
    plt.show()

def plot_eo_threshold(tpr, tnr, S_grid, sensitive_var_name, 
                      dataset_name, target_name,
                      tpr_color='orange', tnr_color='forestgreen'):
    """
    Plot Equality of Odds (EO) (True Positive Rate (TPR) or Recall and True Negative Rate (TNR))
    and its corresponding confidence interval for ground truth and predictions 
    (calculated for continuous sensitive variable with threshold approach).
    
    Parameters
    ----------
    tpr : numpy array
        True Positive Rate (TPR) or Recall.
    tnr : numpy array
        True Negative Rate (TNR).
    S_grid : numpy array
        Discretization of continuous sensitive variable on which TPR, TNR are calculated.
    sensitive_var_name : str
        Name of sensitive variable for x-axis.
    dataset_name : str
        Name of the dataset for plot title.
    target_name : str
        Name of target variable for legend.
    tpr_color : str
        Color for TPR plot.
    tnr_pred_color : str
        Color for TNR plot.
    """
    plt.title(f'Equality of Odds as function of threshold of ${sensitive_var_name}$ ({dataset_name})')
    plt.plot(S_grid, tpr[:, 0], color=tpr_color, label=f'TPR, {target_name}')
    plt.fill_between(S_grid, tpr[:, 1], tpr[:, 2], color=tpr_color, alpha=.2)
    plt.plot(S_grid, tnr[:, 0], color=tnr_color, label=f'TNR, {target_name}')
    plt.fill_between(S_grid, tnr[:, 1], tnr[:, 2], color=tnr_color, alpha=.2)
    plt.xlabel(f'${sensitive_var_name}$')
    plt.legend()
    plt.show()

def plot_conditional_proba(proba, pred_proba, mean_counts, S_grid,  
                           sensitive_var_name, dataset_name, 
                           proba_color='mediumpurple', 
                           pred_proba_color='gold'):
    """
    Plot conditional distribution of ground truth or predictions given sensitive variable
    and its corresponding confidence interval.
    
    Parameters
    ----------
    proba : numpy array
        Conditional distribution of ground truth given sensitive variable.
    pred_proba : numpy array
        Conditional distribution of predictions given sensitive variable.
    mean_counts : numpy array
        Mean number of observations w.r.t. sensitive variable.
    S_grid : numpy array
        Discretization of continuous sensitive variable on which distributions are calculated.
    sensitive_var_name : str
        Name of sensitive variable for x-axis.
    dataset_name : str
        Name of the dataset for plot title.
    proba_color : str
        Color for plot of conditional distribution of ground truth.
    pred_proba_color : str
        Color for plot of conditional distribution of predictions.
    """
    plt.plot(S_grid, proba, label='$P(Y=1|L)$, ground truth', color=proba_color)
    plt.plot(S_grid, pred_proba[:, 0], label='$P(\hat{Y}=1|L)$, prediction', color=pred_proba_color)
    plt.fill_between(S_grid, pred_proba[:, 1], pred_proba[:, 2], color=pred_proba_color, alpha=.2)
    plt.plot(S_grid, mean_counts, label='Mean # of obs in bin', color=proba_color, linestyle='--')
    plt.legend()
    plt.title(dataset_name)