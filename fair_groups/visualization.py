import numpy as np
import matplotlib.pyplot as plt


def plot_partition(partition, phi_by_group, sensitive_var_name="ITA"):
    for i in range(len(partition)):
        plt.axvline(x=partition[i], color="black", linestyle="dashed")

    for i in range(len(partition) - 1):
        plt.hlines(y=phi_by_group[i], xmin=partition[i], xmax=partition[i + 1], color="red")

    plt.xlabel(f"${sensitive_var_name}$")
    plt.ylabel(rf"$\Phi({sensitive_var_name}^{{\mathcal{{P}}}})$")
    plt.show()


def plot_partition_with_ci(partition, partition_ci, sensitive_var_name="ITA"):
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
    plt.plot(s_bins[:-1], y_s_proba)
    plt.xlabel(f"${sensitive_var_name}$")
    plt.ylabel(rf"$P(Y = 1 | {sensitive_var_name})$")
    plt.show()


def plot_group_summary_statistics_table(s, y, partition, phi_by_group, sensitive_var_name="ITA"):
    # Summary table
    summary_data = []
    for i in range(len(partition) - 1):
        mask = (s >= partition[i]) & (s <= partition[i + 1])
        group_size = np.sum(mask)
        group_rate = y[mask].mean()
        summary_data.append([f'Group {i + 1}', f'{partition[i]:.1f}-{partition[i+1]:.1f}', 
                            group_size, f'{group_rate:.3f}', f'{phi_by_group[i]:.4f}'])

    table = plt.table(cellText=summary_data, 
                      colLabels=['Group',  f'${sensitive_var_name}$ Range', 'Size', f'$P(Y=1|{sensitive_var_name})$', '$\Phi$'],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Group Summary Statistics')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
