import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from prefsampling.tree.schroeder import schroeder_tree
from validation.validator import EmptyValidator


def schroeder_fixed_k(num_obs: int, num_leaves: int, k: int, plot_dir_root=None):
    if k > num_leaves - 1:
        raise ValueError("The number of internal nodes k cannot exceed the number of leaves - 1")
    c = Counter()
    for _ in range(num_obs):
        tree = schroeder_tree(num_leaves, num_internal_nodes=k)
        c[tree.anonymous_tree_representation()] += 1

    v = EmptyValidator(num_leaves)
    v.observed_distribution = np.fromiter(c.values(), dtype=float)
    v.observed_distribution /= v.observed_distribution.sum()
    if plot_dir_root is None:
        file_path = None
    else:
        file_path = os.path.join(plot_dir_root, f"schroder_tree_{num_leaves}_{k}.png")
    return v.plot_frequencies(
        f"Distribution of Schröder trees\n#leaves={num_leaves}, #internal nodes={k}\n#observation={num_obs}",
        xlabel="Tree identifier",
        ordering="observed",
        file_path=file_path,
        x_tick_labels=list(c)
    )


def schroeder_multiple_k(num_obs: int, num_leaves: int, all_k: list[int] = None, plot_dir_root=None):
    if all_k is None:
        all_k = list(range(1, num_leaves))
    if all(k > num_leaves - 1 for k in all_k):
        raise ValueError(
            "The number of internal nodes k cannot exceed the number of leaves - 1")
    for k in all_k:
        schroeder_fixed_k(num_obs, num_leaves, k, plot_dir_root)

