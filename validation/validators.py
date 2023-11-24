import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chisquare


def run_chi_square_test(frequencies: np.ndarray, distribution: np.ndarray):
    test_result = chisquare(f_obs=frequencies, f_exp=distribution)
    if test_result.pvalue < 0.1:
        logging.error(f"\tChi-Square test failed, p-value is {test_result.pvalue}.")
        logging.error(
            f"\tThe observed frequencies likely do not follow the theoretical "
            "distribution."
        )
    else:
        logging.info(f"\tChi-Square test passed, p-value is {test_result.pvalue}.")
        logging.info(
            "\tWe cannot conclude that the observations do not come from the theoretical "
            "distribution."
        )
    return test_result


def plot_frequencies(
    frequencies,
    distribution,
    graph_title="",
    graph_sub_title="",
    xlabel="",
    ylabel="",
    x_tick_labels=None,
    file_path=None,
):
    plt.close("all")

    sort_permutation = np.flip(distribution.argsort())
    frequencies = frequencies[sort_permutation]
    distribution = distribution[sort_permutation]
    if x_tick_labels:
        new_x_tick_labels = [x_tick_labels[k] for k in sort_permutation]
        x_tick_labels = new_x_tick_labels

    fig, ax = plt.subplots()
    ax.bar(
        np.arange(len(frequencies)) - 0.2, frequencies, width=0.4
    )
    ax.bar(
        np.arange(len(distribution)) + 0.2, distribution, width=0.4
    )
    if not graph_title:
        graph_title = (
            "Frequencies of the observations compared to the theoretical distribution"
        )
    if graph_sub_title:
        graph_title += "\n" + graph_sub_title
    ax.set_title(graph_title)
    ax.legend(['Observations', 'Theoretical'])
    if not xlabel:
        xlabel = "Rank identifier (ordered by theoretical frequency)"
    ax.set_xlabel(xlabel)
    if not ylabel:
        ylabel = "Frequency"
    ax.set_ylabel(ylabel)
    if x_tick_labels:
        ax.set_xticks(range(len(x_tick_labels)))
        ax.set_xticklabels(x_tick_labels, rotation=90)

    if file_path:
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
