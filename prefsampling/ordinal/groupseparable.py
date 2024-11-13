from __future__ import annotations

import warnings
from enum import Enum
from itertools import chain

import numpy as np

from prefsampling.tree.schroeder import (
    schroeder_tree,
    schroeder_tree_brute_force,
    schroeder_tree_lescanne,
)
from prefsampling.tree.balanced import balanced_tree
from prefsampling.tree.caterpillar import caterpillar_tree
from prefsampling.inputvalidators import validate_num_voters_candidates
from prefsampling.combinatorics import comb


class TreeSampler(Enum):
    """
    Constants use to represent different samplers for trees that can be used for group separable
    preferences.
    """

    SCHROEDER = "Scröder Tree by Alsonso, Rémy, Schott"
    """
    Random Schröder trees sampled following Alonso, Rémy, Schott (1997)
    """

    SCHROEDER_UNIFORM = "Scröder Tree by brute-force"
    """
    Random Schröder sampled uniformly via complete enumeration algorithm
    """

    SCHROEDER_LESCANNE = "Scröder Tree by Lescanne"
    """
    Random Schröder trees sampled following Lescanne (2022)
    """

    CATERPILLAR = "Caterpillar Tree"
    """
    Caterpillar trees
    """

    BALANCED = "Balanced Tree"
    """
    Balanced trees
    """


@validate_num_voters_candidates
def group_separable(
    num_voters: int,
    num_candidates: int,
    tree_sampler: TreeSampler = TreeSampler.SCHROEDER,
    seed: int = None,
) -> list[list[int]]:
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
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import group_separable, TreeSampler

            # Sample a group-separable profile model with 2 voters and 3 candidates using the
            # sampler for Schröder trees proposed by Alonso, Rémy, Schott (1997)
            group_separable(2, 3, tree_sampler=TreeSampler.SCHROEDER)

            # For reproducibility, you can set the seed.
            group_separable(2, 3, tree_sampler=TreeSampler.SCHROEDER_UNIFORM, seed=1002)

    Validation
    ----------

        When using a sampler for trees that is uniform, the probability distribution over
        group-separable profiles generated by this sampler should be uniform too.


    References
    ----------
        `The complexity of election problems with group-separable preferences
        <https://link.springer.com/article/10.1007/s10458-022-09549-7>`_,
        *Faliszewski, Piotr, Alexander Karpov, and Svetlana Obraztsova*,
        Autonomous Agents and Multi-Agent Systems 36:18, 2022.
    """
    rng = np.random.default_rng(seed)

    if isinstance(tree_sampler, Enum):
        tree_sampler = TreeSampler(tree_sampler.value)
    else:
        tree_sampler = TreeSampler(tree_sampler)
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

        if not (buckets >= 0).all():
            warnings.warn(
                "Something went wrong when computing the distribution of the number of "
                "group-separable profiles, probably due to way too high numbers. We are "
                "defaulting to a uniform choice over the number of internal nodes.",
                RuntimeWarning,
            )
            buckets = np.ones(num_candidates - 1)
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
    num_internal_nodes = len(all_inner_nodes)

    votes = []
    frontier = _sample_a_vote(tree_root)
    first_vote = rng.permutation(range(num_candidates))
    vote_map = {j: first_vote[i] for i, j in enumerate(frontier)}
    votes.append(list(first_vote))

    # We sample the signatures, ensuring that there is always at least one "True" per position
    signatures = np.zeros((num_voters - 1, num_internal_nodes), dtype=bool)
    for r in range(num_internal_nodes):
        values_at_pos = rng.choice((True, False), size=num_voters - 1)
        while r > 0 and not any(values_at_pos) and num_voters - 1 > 0:
            values_at_pos = rng.choice((True, False), size=num_voters - 1)
        for i in range(num_voters - 1):
            signatures[i][r] = values_at_pos[i]

    for i in range(1, num_voters):
        signature = signatures[i - 1]

        for j, node in enumerate(all_inner_nodes):
            node.reverse = signature[j]

        raw_vote = _sample_a_vote(tree_root)
        votes.append([vote_map[candidate] for candidate in raw_vote])

    # if tuple(first_vote) == (0, 2, 1) and tree_root.anonymous_tree_representation() == "2(2(_, _), _)":
    #     print(f"t: {tree_root.anonymous_tree_representation()}\nf = {frontier}\nv = {first_vote}\nmap = {vote_map}\nsig:{signatures}\nvotes:\n{votes}\n")

    return votes


def _number_group_separable_profiles(m: int, r: int, n: int) -> int:
    """
    Returns the number of group separable profiles given the number of candidates `m`, the number
    of voters `n` and the number of internal nodes `r`.
    """
    return int(comb(m - 1, r) * comb(m - 1 + r, m) * (2 ** (n - 1) - 1) ** (r - 1))


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
