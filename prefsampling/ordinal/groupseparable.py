import math
from enum import Enum
from itertools import chain

import numpy as np

from prefsampling.tree.schroeder import (
    schroeder_tree,
    schroeder_tree_brute_force,
    schroeder_tree_lescanne,
)
from prefsampling.tree.caterpillar import caterpillar_tree
from prefsampling.tree.balanced import balanced_tree
from prefsampling.inputvalidators import validate_num_voters_candidates


class TreeSampler(Enum):
    """
    Constants use to represent different samplers for trees that can be used for group separable
    preferences.
    """

    SCHROEDER = 1
    """
    Random Schröder trees sampled following Alonso, Rémy, Schott (1997)
    """

    SCHROEDER_UNIFORM = 2
    """
    Random Schröder sampled uniformly via complete enumeration algorithm
    """

    SCHROEDER_LESCANNE = 3
    """
    Random Schröder trees sampled following Lescanne (2022)
    """

    CATERPILLAR = 4
    """
    Caterpillar trees
    """

    BALANCED = 5
    """
    Balanced trees
    """


@validate_num_voters_candidates
def group_separable(
    num_voters: int,
    num_candidates: int,
    tree_sampler: TreeSampler = TreeSampler.SCHROEDER,
    seed: int = None,
):
    """
    Samplers for group separable votes. For the definition of group-seprable preferences, see
    `Elkind, Lackner, Peters (2022) <https://arxiv.org/abs/2205.09092>`_.

    This sampler implements the algorithm presented by `Faliszewski, Karpov, Obraztsova (2022)
    <https://link.springer.com/article/10.1007/s10458-022-09549-7>`_. The implementation follows
    these general steps. First, a decomposition tree is sampled at random. Then, for each internal
    node of the tree the order of its children is reversed with probability 0.5. The vote then
    corresponds to the label of the leaves of the tree read from left to right.

    This sampler only generates neutral collections of votes. This means that the first vote is
    always `0 > 1 > 2 > ...`.

    When used with a uniform sampler over all decomposition trees it is meant to yield a uniform
    distribution over neutral group separable collections of votes. However, our analysis indicates
    that it is not the case: collections of vote containing only `0 < 1 < 2 < ... ` or
    `m-1 < m-2 < m-3 < ...` are over-represented.

    Note that for a given number of voters, votes are sampled independently but the number of voters
    can impact the sampling of the decomposition tree.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        tree_sampler : TreeSampler, default: :py:class:`~prefsampling.ordinal.groupseparable.TreeSampler.SCHROEDER`
            Sampler used to sample the tree. Should be one of the constants defined in the
            :py:class:`~prefsampling.ordinal.groupseparable.TreeSampler` enumeration.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """

    rng = np.random.default_rng(seed)

    if tree_sampler in (
        TreeSampler.SCHROEDER,
        TreeSampler.SCHROEDER_UNIFORM,
        TreeSampler.SCHROEDER_LESCANNE,
    ):
        buckets = np.zeros(num_candidates - 1)
        for r in range(1, num_candidates):
            buckets[r - 1] = _number_group_separable_profiles(
                num_candidates, r, num_voters
            )
        buckets /= buckets.sum()
        num_internal_nodes = rng.choice(len(buckets), p=buckets) + 1

        if tree_sampler == TreeSampler.SCHROEDER:
            tree_root = schroeder_tree(num_candidates, num_internal_nodes, seed)
        elif tree_sampler == TreeSampler.SCHROEDER_UNIFORM:
            tree_root = schroeder_tree_brute_force(
                num_candidates, num_internal_nodes, seed
            )
        elif tree_sampler == TreeSampler.SCHROEDER_LESCANNE:
            tree_root = schroeder_tree_lescanne(
                num_candidates, num_internal_nodes, seed
            )
        else:
            raise ValueError("There is something weird with the tree_sampler value...")
        tree_root.rename_frontier()
    elif tree_sampler == TreeSampler.CATERPILLAR:
        tree_root = caterpillar_tree(num_candidates)

    elif tree_sampler == TreeSampler.BALANCED:
        tree_root = balanced_tree(num_candidates)
    else:
        raise ValueError(
            "The `tree` argument needs to be one of the constant defined in the "
            "ordinal.TreeSampler enumeration. Choices are: "
            + ", ".join(str(s) for s in TreeSampler)
        )

    all_inner_nodes = tree_root.internal_nodes()

    votes = np.zeros((num_voters, num_candidates), dtype=int)
    frontier = _sample_a_vote(tree_root)
    vote_map = {j: j for i, j in enumerate(frontier)}
    votes[0] = [vote_map[candidate] for candidate in frontier]
    for i in range(1, num_voters):
        signature = rng.choice((True, False), size=len(all_inner_nodes))

        for j, node in enumerate(all_inner_nodes):
            node.reverse = signature[j]

        raw_vote = _sample_a_vote(tree_root)
        votes[i] = tuple(vote_map[candidate] for candidate in raw_vote)

        for j, node in enumerate(all_inner_nodes):
            node.reverse = False

    return votes


def _number_group_separable_profiles(m: int, r: int, n: int) -> float:
    """
    Returns the number of decomposition trees given the number of candidates `m` and the number
    of internal nodes `r` based on the formula from `Karpov (2019)
    <https://link.springer.com/article/10.1007/s10726-019-09621-w>`_
    """
    return math.comb(m - 1, r) * math.comb(m - 1 + r, m) * (2 ** (n - 1) - 1) ** (r - 1)


def _sample_a_vote(node, reverse=False):
    """
    Recursively samples a single vote.
    """
    if node.leaf:
        return [node.identifier]
    output = []
    if reverse == node.reverse:
        for c in node.children:
            output.append(_sample_a_vote(c))
    else:
        for i in range(len(node.children) - 1, -1, -1):
            output.append(_sample_a_vote(node.children[i], reverse=True))

    return list(chain.from_iterable(output))
