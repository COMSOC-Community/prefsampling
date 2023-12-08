import math
from enum import Enum
from itertools import chain

import numpy as np

from prefsampling.tree.schroeder import schroeder_tree
from prefsampling.tree.caterpillar import generate_caterpillar_tree
from prefsampling.tree.balanced import generate_balanced_tree
from prefsampling.inputvalidators import validate_num_voters_candidates


class DecompositionTree(Enum):
    """
    Constants use to represent different types of decomposition trees for group separable
    preferences.
    """

    RANDOM = 1
    """
    Random trees
    """

    CATERPILLAR = 2
    """
    Caterpillar trees
    """

    BALANCED = 3
    """
    Balanced trees
    """


@validate_num_voters_candidates
def group_separable(
    num_voters: int,
        num_candidates: int,
        tree: DecompositionTree = DecompositionTree.RANDOM,
        seed: int = None
):
    """
    Samplers for group separable votes. For the definition of group-seprable preferences, see
    `Elkind, Lackner, Peters (2022) <https://arxiv.org/abs/2205.09092>`_.

    This sampler yields a uniform distribution over group separable collections of votes. It
    implements the algorithm presented by `Faliszewski, Karpov, Obraztsova (2022)
    <https://link.springer.com/article/10.1007/s10458-022-09549-7>`_

    Note that for a given number of voters, votes are not sampled independently.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        tree : DecompositionTree, default: :py:class:`~prefsampling.ordinal.groupseparable.DecompositionTree.RANDOM`
            Type of decomposition tree considered. Should be a constant defined in the
            :py:class:`~prefsampling.ordinal.groupseparable.DecompositionTree` enumeration.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """

    rng = np.random.default_rng(seed)

    if tree == DecompositionTree.RANDOM:
        buckets = np.zeros(num_candidates - 1)
        for r in range(1, num_candidates):
            buckets[r - 1] = _number_decomposition_tree(num_candidates, r, num_voters)
        buckets /= buckets.sum()
        num_internal_nodes = rng.choice(len(buckets), p=buckets) + 1

        tree_root = generate_schroder_tree(
            num_candidates, num_internal_nodes, seed
        )
    elif tree == DecompositionTree.CATERPILLAR:
        tree_root = generate_caterpillar_tree(num_candidates)

    elif tree == DecompositionTree.BALANCED:
        tree_root = generate_balanced_tree(num_candidates)
    else:
        raise ValueError(
            "The `tree` argument needs to be one of the constant defined in the "
            "ordinal.DecompositionTree enumeration. Choices are: "
            + ", ".join(str(s) for s in DecompositionTree)
        )

    all_inner_nodes = tree_root.get_all_inner_nodes()

    votes = np.zeros((num_voters, num_candidates), dtype=int)
    frontier = _sample_a_vote(tree_root)
    vote_map = {i: j for i, j in enumerate(frontier)}
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


def _number_decomposition_tree(m: int, r: int, n=1) -> float:
    """
    Returns the number of decomposition trees given the number of candidates `m` and the number
    of internal nodes `r` based on the formula from `Karpov (2019)
    <https://link.springer.com/article/10.1007/s10726-019-09621-w>`_
    """
    return math.comb(m - 1, r) * math.comb(m - 1 + r, m) * (2**(n-1) - 1)**(r-1)


def _sample_a_vote(node, reverse=False):
    """
    Recursively samples a single vote.
    """
    if node.leaf:
        return [node.element_id]
    output = []
    if reverse == node.reverse:
        for c in node.children:
            output.append(_sample_a_vote(c))
    else:
        for i in range(len(node.children) - 1, -1, -1):
            output.append(
                _sample_a_vote(node.children[i], reverse=True)
            )

    return list(chain.from_iterable(output))

