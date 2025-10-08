import numpy as np
import matplotlib.pyplot as plt


def plot_partition(partition, phi_by_group, sensitive_var_name="ITA"):
    """
    Plots the partition of the sensitive variable and the corresponding phi values for each group.
    Parameters:
    ----------
    partition: list or array
        List or array of partition points for the sensitive variable.
    phi_by_group: list or array
        List or array of phi values corresponding to each group defined by the partition.
    sensitive_var_name: str
        Name of the sensitive variable (default is "ITA").
    Returns:
    ----------
    A matplotlib plot showing the partition and phi values.
    """
    for i in range(len(partition)):
        plt.axvline(x=partition[i], color="black", linestyle="dashed")

    for i in range(len(partition) - 1):
        plt.hlines(y=phi_by_group[i], xmin=partition[i], xmax=partition[i + 1], color="red")

    plt.xlabel(f"${sensitive_var_name}$")
    plt.ylabel(rf"$\Phi({sensitive_var_name}^{{\mathcal{{P}}}})$")
    plt.show()


def plot_partition_with_ci(partition, partition_ci, sensitive_var_name="ITA"):
    """
    Plots the partition of the sensitive variable and the corresponding phi values with confidence intervals for each group.
    Parameters:
    ----------
    partition: list or array
        List or array of partition points for the sensitive variable.
    partition_ci: array
        Array of shape (n_groups, 3) where each row contains [lower_bound, upper_bound, mean] for the phi value of each group.
    sensitive_var_name: str
        Name of the sensitive variable (default is "ITA").
    Returns:
    ----------
    A matplotlib plot showing the partition and phi values with confidence intervals.
    """
    for i in range(len(partition)):
        plt.axvline(x=partition[i], color="black", linestyle="dashed")

    for i in range(len(partition) - 1):
        plt.hlines(y=partition_ci[i, 0], xmin=partition[i], xmax=partition[i + 1], color="red")
        plt.fill_between(
            x=[partition[i], partition[i + 1]],
            y1=partition_ci[i, 2],
            y2=partition_ci[i, 1],
            color="orange",
            interpolate=True,
            alpha=0.5,
        )

    plt.xlabel(f"${sensitive_var_name}$")
    plt.ylabel(rf"$\Phi({sensitive_var_name}^{{\mathcal{{P}}}})$")
    plt.show()


def plot_conditional_proba(s_bins, y_s_proba, sensitive_var_name="ITA"):
    """
    Plots the conditional probability P(Y=1 | sensitive variable) against the sensitive variable.
    Parameters:
    ----------
    s_bins: array
        Array of bin edges for the sensitive variable.
    y_s_proba: array
        Array of conditional probabilities P(Y=1 | sensitive variable) for each bin.
    sensitive_var_name: str
        Name of the sensitive variable (default is "ITA").
    Returns:
    ----------
    A matplotlib plot showing the conditional probabilities.
    """
    plt.plot(s_bins[:-1], y_s_proba)
    plt.xlabel(f"${sensitive_var_name}$")
    plt.ylabel(rf"$P(Y = 1 | {sensitive_var_name})$")
    plt.show()


def plot_group_summary_statistics_table(s, y, partition, phi_by_group, sensitive_var_name="ITA"):
    """
    Plots a summary statistics table for each group defined by the partition of the sensitive variable.
    Parameters:
    ----------
    s: array
        Array of sensitive variable values.
    y: array
        Array of binary outcome values (0 or 1).
    partition: list or array
        List or array of partition points for the sensitive variable.
    phi_by_group: list or array
        List or array of phi values corresponding to each group defined by the partition.
    sensitive_var_name: str
        Name of the sensitive variable (default is "ITA").
    Returns:
    ----------
    A matplotlib plot showing the summary statistics table.
    """
    # Summary table
    summary_data = []
    for i in range(len(partition) - 1):
        mask = (s >= partition[i]) & (s <= partition[i + 1])
        group_size = np.sum(mask)
        group_rate = y[mask].mean()
        summary_data.append(
            [
                f"Group {i + 1}",
                f"{partition[i]:.1f}-{partition[i + 1]:.1f}",
                group_size,
                f"{group_rate:.3f}",
                f"{phi_by_group[i]:.4f}",
            ]
        )

    table = plt.table(
        cellText=summary_data,
        colLabels=[
            "Group",
            f"${sensitive_var_name}$ Range",
            "Size",
            f"$P(Y=1|{sensitive_var_name})$",
            r"$\Phi$",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title("Group Summary Statistics")
    plt.tight_layout()
    plt.axis("off")
    plt.show()
