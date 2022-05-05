import numpy as np
import seaborn as sns


# --------plot figs--------
def plot_pre_scatter(true, pre, save_pth):
    min_ = np.min(true)
    max_ = np.max(true)

    red_dot_x = true[np.abs(true-pre)>0.5]
    red_dot_y = pre[np.abs(true-pre)>0.5]

    nm_dot_x = true[np.abs(true-pre)<0.5]
    nm_dot_y = pre[np.abs(true-pre)<0.5]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(nm_dot_x, nm_dot_y, edgecolors=None, alpha=0.6)
    sns.scatterplot(red_dot_x, red_dot_y, color='red', edgecolors=None, alpha=0.6)
    sns.lineplot([min_, max_], [min_, max_], linestyle='--', color='black')

    plt.xlabel(f'cycloplegic SE')
    plt.ylabel(f'predictions')
    plt.tick_params()
    sns.despine()
    plt.savefig(save_pth)
    plt.show()

