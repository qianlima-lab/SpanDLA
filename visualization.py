import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

matplotlib.use('Agg')


# data = np.random.uniform(0, 1, size=(10, 10))


def plt_heatmap(data, path):
    f, ax = plt.subplots()
    x = list(range(0, data.shape[0]))
    ax.xaxis.tick_top()
    sns.heatmap(data, cmap='GnBu', yticklabels=['s' + str(i + 1) for i in x], xticklabels=['s' + str(i + 1) for i in x])
    plt.savefig(path, bbox_inches='tight')
    plt.close(f)

