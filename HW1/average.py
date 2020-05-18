import json

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import pandas as pd

experiments = []


def plotting(vals, function):
    plt.figure()

    X = [exp for i, exp in vals for _ in range(len(i))]
    Y = [i[j] for i, _ in vals for j in range(len(i))]
    sns.boxplot(x=X, y=Y)
    plt.xlabel("Experiment values", fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.show()
    plt.savefig(f'../raport/boxplots/{function}_boxplot.eps', format='eps')


def plot_experiments(function):
    exps = ["1a", "1b", "2a", "2b", "3a"]
    for exp in exps:
        file_hc = f"../raport/experiment_{exp}_{function}/best_values_hc.json"
        file_ga = f"../raport/experiment_{exp}_{function}/best_values_ga.json"
        try:
            fp = open(file_hc, "r")
        except:
            fp = open(file_ga, "r")

        vals = json.loads(json.load(fp))
        experiments.append((vals, exp))
    # plotting(experiments, function)
    mat = [[0 for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(0, 5):
            mat[i][j] = scipy.stats.ttest_ind(experiments[i][0], experiments[j][0]).pvalue

    t_test = pd.DataFrame(mat)

    fig, ax = plt.subplots()
    ax.xaxis.tick_top()
    sns.heatmap(t_test, annot=True, fmt='g',
                cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
    ax.set_yticklabels(exps, rotation="horizontal")
    ax.set_xticklabels(exps)
    plt.savefig(f'../raport/t_test/{function}_t_test_matrix.png', bbox_inches='tight', pad_inches=0.0)


# ["rosenbrock", "rastrigin", "griewangk"]:
plot_experiments("griewangk")

# vals = json.load(open(f"../raport/experiment_3a_rosenbrock/best_values_hc.json", "r"))
# vals_list = json.loads(vals)
# print(f"average - {np.average(vals_list)}", "min", min(vals_list), "max", max(vals_list))
# experiments.append(vals_list)
#
# print(np.array(experiments))
