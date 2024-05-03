"""
A collection of functions used to iterable over all possible rankings, profiles, etc...
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from copy import deepcopy
from itertools import chain, combinations, permutations, combinations_with_replacement

import numpy as np

from prefsampling.inputvalidators import validate_int


def comb(n: int, k: int) -> int:
    """
    Function to compute the binomial coefficient. It uses math.comb if available (i.e., if Python >=
    3.8), otherwise computes it by hand.

    Parameters
    ----------
        n: int
            n in `n chooses k`, i.e., the size of the whole set
        k: int
            k in `n chooses k`, i.e.; the size of the subset

    Returns
    -------
        int
            The value of n chooses k
    """
    if hasattr(math, "comb"):
        return math.comb(n, k)
    return _comb(n, k)


def _comb(n: int, k: int) -> int:
    """
    Computation of `n chooses k`.

    Parameters
    ----------
        n: int
            n in `n chooses k`, i.e., the size of the whole set
        k: int
            k in `n chooses k`, i.e.; the size of the subset

    Returns
    -------
        int
            The value of n chooses k
    """
    try:
        return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
    except ValueError:
        return 0


def generalised_ascending_factorial(value: int, length: int, increment: float) -> float:
    """
    Computes the ascending factorial. The ascending factorial is equal to:
    :math:`\\text{value} \\times (\\text{value} + \\text{increment}) \\times \\ldots \\times
    (\\text{value} + (\\text{length} - 1) \\times \\text{increment})`.

    Parameters
    ----------
        value: int
            The value whose generalised ascending factorial we are computing.
        length: int
            The number of terms in the ascending factorial to compute.
        increment: float
            The increment to add to the value in each term.

    Returns
    -------
        float
            The value of the (generalised) ascending factorial.

    """
    validate_int(value, "value", lower_bound=0)
    validate_int(value, "length", lower_bound=0)
    if length == 0:
        return 1
    return (value + (length - 1) * increment) * generalised_ascending_factorial(
        value, length - 1, increment
    )


def powerset(
    iterable: Iterable, min_size: int = 1, max_size: int = None
) -> tuple[tuple]:
    """
    Returns the powerset of the iterable.

    Parameters
    ----------
        iterable: Iterable
            The iterable.
        min_size: int, default: :code:`1`
            Any subset of size smaller than `min_size` is excluded.
        max_size: int, default: :code:`len(iterable)`
            Any subset of size larger than `max_size` is excluded.

    Returns
    -------
        tuple[tuple]
            The powerset of the input iterable.
    """
    s = list(iterable)
    if max_size is None:
        max_size = len(s) + 1
    return tuple(
        tuple(sorted(s))
        for s in chain.from_iterable(
            combinations(s, r) for r in range(min_size, max_size)
        )
    )


def proper_powerset(iterable: Iterable, min_size: int = 1) -> tuple[tuple]:
    """
    Returns the set of all proper subsets of the iterable.

    Parameters
    ----------
        iterable: Iterable
            The iterable.
        min_size: int, default: :code:`1`
            Any subset of size smaller than `min_size` is excluded.

    Returns
    -------
        tuple[tuple]
            The set of all subsets of the input iterable.
    """
    s = list(iterable)
    return powerset(s, min_size=min_size, max_size=len(s))


def all_rankings(num_elements: int) -> list[tuple[int]]:
    """
    Returns a list of all the rankings with `num_elements`.

    Parameters
    ----------
        num_elements: int
            The number of elements.

    Returns
    -------
        list[tuple[int]]
            The list of all the rankings with `num_elements`, each ranking being represented as a
            tuple of int.

    """
    return [tuple(rank) for rank in permutations(range(num_elements))]


def all_anonymous_profiles(
    num_voters: int, num_candidates: int
) -> list[tuple[tuple[int]]]:
    """
    Returns a list of all the anonymous profiles for a given number of voters and candidates. An
    anonymous profile is a sorted tuple of rankings, each ranking being represented by a tuple of
    int. It is anonymous in the sense that the position of the voters does not matter. We implement
    this by sorting the rankings in the profile in lexicographic order.

    Parameters
    ----------
        num_voters: int
            The number of voters
        num_candidates: int
            The number of candidates (the length of the rankings)

    Returns
    -------
        list[tuple[tuple[int]]]
            A list containing all the anonymous profiles
    """
    return list(combinations_with_replacement(all_rankings(num_candidates), num_voters))


def all_profiles(num_voters: int, num_candidates: int) -> list[tuple[tuple[int]]]:
    """
    Returns a list of all the profiles for a given number of voters and candidates. We compute this
    by considering all permutations of all the anonymous profiles.

    Parameters
    ----------
        num_voters: int
            The number of voters
        num_candidates: int
            The number of candidates (the length of the rankings)

    Returns
    -------
        list[tuple[tuple[int]]]
            A list containing all the profiles
    """
    res = set()
    for prof in all_anonymous_profiles(num_voters, num_candidates):
        for p in permutations(prof):
            res.add(tuple(p))
    return list(res)


def all_non_isomorphic_profiles(
    num_voters: int,
    num_candidates: int,
    profiles: Iterable[Sequence[Sequence[int]]] = None,
) -> list[tuple[tuple[int]]]:
    """
    Returns a maximal collection of profiles that are not isomorphic. Two profiles are isomorphic
    if there exists a renaming of the candidates such that the two profiles are the same.

    Parameters
    ----------
        num_voters: int
            The number of voters
        num_candidates: int
            The number of candidates (the length of the rankings)
        profiles: Sequence[Sequence[int]], defaults: :code:`None`
            The collection of all profiles. If the argument is not provided,
            :py:func:`prefsampling.combinatorics.all_profiles` is used.

    Returns
    -------
        list[tuple[tuple[int]]]
            A list containing all the profiles, selecting only one single profile per equivalence
            class.
    """
    res = set()
    if profiles is None:
        profiles = all_profiles(num_voters, num_candidates)
    for profile in profiles:
        perm = tuple(profile[0])
        res.add(tuple(tuple(perm.index(c) for c in r) for r in profile))
    return list(res)


def all_single_peaked_rankings(num_candidates: int) -> list[tuple[int]]:
    """
    Returns the list of all the single-peaked rankings with respect to the axis `0, 1, ..., m - 1`.

    Parameters
    ----------
        num_candidates: int
            The number of candidates (the length of the rankings)

    Returns
    -------
        list[tuple[int]]
            A list containing all single-peaked rankings.
    """

    def recursive(a, b, all_sp_ranks, rank, position):
        if a == b:
            rank[position] = a
            all_sp_ranks.append(tuple(rank))
            return
        rank[position] = a
        recursive(a + 1, b, all_sp_ranks, rank, position - 1)

        rank = deepcopy(rank)
        rank[position] = b
        recursive(a, b - 1, all_sp_ranks, rank, position - 1)

    res = []
    recursive(0, num_candidates - 1, res, [0] * num_candidates, num_candidates - 1)
    return res


def all_single_peaked_circle_rankings(num_candidates: int):
    """
    Returns the list of all the single-peaked rankings with respect to the circular axis
    `0, 1, ..., m - 1, 0`.

    Parameters
    ----------
        num_candidates: int
            The number of candidates (the length of the rankings)

    Returns
    -------
        list[tuple[int]]
            A list containing all single-peaked on a circle rankings.
    """

    def recursive(a, b, all_sp_ranks, rank, position):
        if a < 0:
            a += num_candidates
        if b > num_candidates - 1:
            b -= num_candidates
        if a == b:
            rank[position] = a
            all_sp_ranks.append(tuple(rank))
            return
        rank[position] = a
        recursive(a - 1, b, all_sp_ranks, rank, position + 1)

        rank = deepcopy(rank)
        rank[position] = b
        recursive(a, b + 1, all_sp_ranks, rank, position + 1)

    res = []
    for peak in range(num_candidates):
        recursive(peak - 1, peak + 1, res, [peak] + [0] * (num_candidates - 1), 1)
    return res


def is_single_crossing(profile: Sequence[Sequence[int]]) -> bool:
    """
    Tests whether a profile is single-crossing given the current ordering of the voters.

    Parameters
    ----------
        profile: Sequence[Sequence[int]]
            The profile

    Returns
    -------
        bool
            True if the profile is single-crossing and false otherwise.
    """
    for j, cand1 in enumerate(profile[0]):
        for cand2 in profile[0][j + 1 :]:
            cand1_over_cand2 = True
            for vote in profile:
                if vote.index(cand1) < vote.index(cand2) and not cand1_over_cand2:
                    return False
                elif vote.index(cand1) > vote.index(cand2):
                    cand1_over_cand2 = False
    return True


def all_single_crossing_profiles(
    num_voters: int,
    num_candidates: int,
    profiles: Iterable[Sequence[Sequence[int]]] = None,
    fix_order: bool = False,
) -> list[tuple[tuple[int]]]:
    """
    Returns all profiles that are single-crossing.

    Parameters
    ----------
        num_voters: int
            The number of voters
        num_candidates: int
            The number of candidates (the length of the rankings)
        profiles: Sequence[Sequence[int]], defaults: :code:`None`
            The collection of all profiles. If the argument is not provided,
            :py:func:`prefsampling.combinatorics.all_profiles` is used.
        fix_order: bool
            If true, only profile that are single-crossing with respect to the current ordering of
            the voters are considered.

    Returns
    -------
        list[tuple[tuple[int]]]
            A list of all the single-crossing profiles.
    """
    if profiles is None:
        profiles = all_profiles(num_voters, num_candidates)
    res = []
    for profile in profiles:
        if fix_order:
            if is_single_crossing(profile):
                res.append(profile)
        else:
            for perm in permutations(profile):
                if is_single_crossing(perm):
                    res.append(profile)
                    break
    return res


class GSNode:
    """
    Class used to represent nodes in a group-separable profiles.
    """

    def __init__(self, cand_set):
        self.cand_set: set[int] = cand_set
        self.children: list[GSNode] = []

    def __repr__(self):
        return f"N{self.cand_set}"

    def tree_representation(self) -> str:
        s = f"{len(self.cand_set)}("
        s += ", ".join(sorted(n.tree_representation() for n in self.children))
        s += ")"
        return s

    def print_tree(self, indent: str = "") -> None:
        print(f"{indent}{self.cand_set}")
        for child in self.children:
            child.print_tree(indent + " ")


def gs_structure(profile: Sequence[Sequence[int]], verbose: bool = False) -> str:
    """
    Computes the group-separable structure for a given profile.

    Parameters
    ----------
        profile: Sequence[Sequence[int]]
            The profile.
        verbose: bool, defaults: :code:`None`
            If True, then additional information is printed.

    Returns
    -------
        str
            A string representing the structure.

    """

    def aux(prof, cands, node):
        if len(cands) == 2:
            new_node = GSNode(tuple(cands))
            node.children.append(new_node)
            return
        if len(cands) == 1:
            return
        found_one_subcands = False
        for j in range(len(cands) - 1, 0, -1):
            subcands = cands[:j]
            if verbose:
                print(f"\tcands={cands}: subcands = {subcands}")
            all_voters_separate = True
            all_voters_separate_above = True
            all_voters_separate_below = True
            for rank in prof:
                sub_cands_indices = set()
                outside_indices = set()
                for i, c in enumerate(rank):
                    if c in subcands:
                        sub_cands_indices.add(i)
                    elif c in cands:
                        outside_indices.add(i)
                if verbose:
                    print(f"\t\trank={rank}: {sub_cands_indices}, {outside_indices}")
                if sub_cands_indices and outside_indices:
                    all_above = True
                    all_below = True
                    for sub_i in sub_cands_indices:
                        for cand_i in outside_indices:
                            if sub_i < cand_i:
                                all_below = False
                            elif sub_i > cand_i:
                                all_above = False
                    if all_above:
                        all_voters_separate_below = False
                    if all_below:
                        all_voters_separate_above = False
                    if not all_above and not all_below:
                        all_voters_separate = False
                        if verbose:
                            print("\t\tBreak, voter fails!")
                        break
            if all_voters_separate:
                if verbose:
                    print("\t\tAll votes separate")
                if j == len(cands) - 1 and (
                    all_voters_separate_below or all_voters_separate_above
                ):
                    new_node = GSNode(tuple(cands))
                    node.children.append(new_node)
                    return
                new_node = GSNode(tuple(subcands))
                node.children.append(new_node)
                aux(prof, subcands, new_node)
                subcands2 = cands[j:]
                new_node2 = GSNode(tuple(subcands2))
                node.children.append(new_node2)
                aux(prof, subcands2, new_node2)
                found_one_subcands = True
                break
        if not found_one_subcands:
            if verbose:
                print(f"\tFor cands={cands} it fails")
            raise ValueError(f"Profile {prof} is not GS")

    if verbose:
        print(profile)
    root = GSNode(tuple(profile[0]))
    aux(profile, profile[0], root)
    return root.tree_representation()


def all_group_separable_profiles(
    num_voters: int,
    num_candidates: int,
    profiles: Iterable[Sequence[Sequence[int]]] = None,
) -> list[tuple[tuple[int]]]:
    """
    Returns all profiles that are group-separable.

    Parameters
    ----------
        num_voters: int
            The number of voters
        num_candidates: int
            The number of candidates (the length of the rankings)
        profiles: Iterable[Sequence[Sequence[int]]], defaults: :code:`None`
            The collection of all profiles. If the argument is not provided,
            :py:func:`prefsampling.combinatorics.all_profiles` is used.

    Returns
    -------
        list[tuple[tuple[int]]]
            A list of all the single-crossing profiles.
    """
    if profiles is None:
        profiles = all_profiles(num_voters, num_candidates)

    res = []
    for profile in profiles:
        # print(profile)
        all_cands_separated = True
        for cands in powerset(range(num_candidates)):
            proper_subsets = list(proper_powerset(cands))
            # print(f"\tcands={cands}: subsets = {proper_subsets}")
            if proper_subsets:
                one_subcands_exists = False
                for subcands in proper_subsets:
                    # print(f"\t\tsubcands={subcands}")
                    all_voters_separate = True
                    for rank in profile:
                        sub_cands_indices = set()
                        outside_indices = set()
                        for i, c in enumerate(rank):
                            if c in subcands:
                                sub_cands_indices.add(i)
                            elif c in cands:
                                outside_indices.add(i)
                        # print(f"\t\trank={rank}: {sub_cands_indices}, {outside_indices}")
                        if sub_cands_indices and outside_indices:
                            all_above = True
                            all_below = True
                            for sub_i in sub_cands_indices:
                                for cand_i in outside_indices:
                                    if sub_i < cand_i:
                                        all_below = False
                                    elif sub_i > cand_i:
                                        all_above = False
                            if not all_above and not all_below:
                                all_voters_separate = False
                                # print("\t\tBreak, voter fails!")
                                break
                    if all_voters_separate:
                        one_subcands_exists = True
                        # print("Break nicely!!")
                        break
                if not one_subcands_exists:
                    all_cands_separated = False
                    # print("Break bdaly!!!")
                    break
        if all_cands_separated:
            res.append(profile)

    # print(res)
    return res


def all_gs_structure(
    num_voters: int = None,
    num_candidates: int = None,
    gs_profiles: Iterable[Sequence[Sequence[int]]] = None,
) -> list[str]:
    """
    Returns the group-separable structures for which at least one profile is group-separable.

    Parameters
    ----------
        num_voters: int
            The number of voters
        num_candidates: int
            The number of candidates (the length of the rankings)
        gs_profiles: Iterable[Sequence[Sequence[int]]], defaults: :code:`None`
            The collection of all group-separable profiles. If the argument is not provided,
            :py:func:`prefsampling.combinatorics.all_group_separable_profiles` is used.

    Returns
    -------
        list[str]
            A list of all the string representations of the structure.

    """
    if gs_profiles is None:
        if num_voters is None and num_candidates is None:
            raise ValueError(
                "You need to provide either number of voters and candidates or a "
                "collection of profiles"
            )
        gs_profiles = all_group_separable_profiles(num_voters, num_candidates)
    return list(set(gs_structure(p) for p in gs_profiles))


def kendall_tau_distance(ranking_1: Iterable, ranking_2: Sequence | np.ndarray) -> int:
    """
    Computes the Kendall-Tau distance between two rankings.

    Parameters
    ----------
        ranking_1: Iterable
            The first ranking
        ranking_2: Sequence | np.ndarray
            The second ranking

    Returns
    -------
        int
            The Kendall-Tau distance between the two rankings.
    """
    distance = 0
    for k, alt1 in enumerate(ranking_1):
        for alt2 in ranking_1[k + 1 :]:
            if ranking_2.index(alt2) < ranking_2.index(alt1):
                distance += 1
    return distance
