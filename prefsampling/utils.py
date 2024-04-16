import math
from collections.abc import Iterable
from itertools import chain, combinations, permutations, combinations_with_replacement


def comb(n: int, k: int):
    """
    Function to compute the binomial coefficient. It uses math.comb if available (i.e., if Python >=
    3.8), otherwise computes it by hand.

    Parameters
    ----------
    n
    k

    Returns
    -------

    """
    if hasattr(math, "comb"):
        return math.comb(n, k)
    return _comb(n, k)


def _comb(n, k):
    try:
        return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
    except ValueError:
        return 0


def powerset(iterable: Iterable, min_size: int = 1, max_size: int = None) -> Iterable:
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
        Iterable
            The powerset of the input iterable.
    """
    s = list(iterable)
    if max_size is None:
        max_size = len(s) + 1
    return chain.from_iterable(combinations(s, r) for r in range(min_size, max_size))


def proper_powerset(iterable, min_size: int = 1):
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
        Iterable
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
    anonymous profile is a sorted tuple of  rankings, each ranking being represented by a tuple of
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
    return list(
        combinations_with_replacement(all_rankings(num_candidates), num_voters)
    )


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
    num_voters: int, num_candidates: int, profiles: tuple[tuple[int]] = None
) -> list[tuple[tuple[int]]]:
    """
    Returns the list of all the non-isomorphic profiles.

    Parameters
    ----------
        num_voters: int
            The number of voters
        num_candidates: int
            The number of candidates (the length of the rankings)
        profiles: tuple[tuple[int]], defaults: :code:`None`
            The collection of all profiles. If the argument is not provided,
            :py:func:`prefsampling.utils.all_profiles` is used.

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


def get_all_single_peaked_ranks(num_candidates: int):
    def recursor(a, b, all_sp_ranks, rank, position):
        if a == b:
            rank[position] = a
            all_sp_ranks.append(tuple(rank))
            return
        rank[position] = a
        recursor(a + 1, b, all_sp_ranks, rank, position - 1)

        rank = deepcopy(rank)
        rank[position] = b
        recursor(a, b - 1, all_sp_ranks, rank, position - 1)

    res = []
    recursor(0, num_candidates - 1, res, [0] * num_candidates, num_candidates - 1)
    return res


def get_all_single_peaked_circle_ranks(num_candidates: int):
    def recursor(a, b, all_sp_ranks, rank, position):
        if a < 0:
            a += num_candidates
        if b > num_candidates - 1:
            b -= num_candidates
        if a == b:
            rank[position] = a
            all_sp_ranks.append(tuple(rank))
            return
        rank[position] = a
        recursor(a - 1, b, all_sp_ranks, rank, position + 1)

        rank = deepcopy(rank)
        rank[position] = b
        recursor(a, b + 1, all_sp_ranks, rank, position + 1)

    res = []
    for peak in range(num_candidates):
        recursor(peak - 1, peak + 1, res, [peak] + [0] * (num_candidates - 1), 1)
    return res


def is_single_crossing(profile):
    for j, cand1 in enumerate(profile[0]):
        for cand2 in profile[0][j + 1 :]:
            cand1_over_cand2 = True
            for vote in profile:
                if vote.index(cand1) < vote.index(cand2) and not cand1_over_cand2:
                    return False
                elif vote.index(cand1) > vote.index(cand2):
                    cand1_over_cand2 = False
    return True


def get_all_single_crossing_profiles(
    num_voters: int,
    num_candidates: int,
    all_profiles=None,
    fix_order=False,
):
    if all_profiles is None:
        all_profiles = get_all_profiles(num_voters, num_candidates)
    res = []
    for profile in all_profiles:
        if fix_order:
            if is_single_crossing(profile):
                res.append(profile)
        else:
            for perm in permutations(profile):
                if is_single_crossing(perm):
                    res.append(profile)
                    break
    return res


def get_all_sc_profiles_non_iso(
    num_voters: int,
    num_candidates: int,
):
    return get_all_single_crossing_profiles(
        num_voters,
        num_candidates,
        all_profiles=get_all_non_isomorphic_profilles(
            num_voters,
            num_candidates,
            all_profiles=get_all_anonymous_profiles(num_voters, num_candidates),
        ),
        fix_order=True,
    )


class GSNode:
    def __init__(self, cand_set):
        self.cand_set = cand_set
        self.children = []

    def __repr__(self):
        return f"N{self.cand_set}"

    def tree_representation(self):
        s = f"{len(self.cand_set)}("
        s += ", ".join(sorted(n.tree_representation() for n in self.children))
        s += ")"
        return s


def print_tree(node, indent=""):
    print(f"{indent}{node.cand_set}")
    for child in node.children:
        print_tree(child, indent + "  ")


def gs_structure(profile, verbose=False):
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


def get_all_group_separable_profiles(
    num_voters: int, num_candidates: int, all_profiles=None
):
    if all_profiles is None:
        all_profiles = get_all_profiles(num_voters, num_candidates)

    res = []
    for profile in all_profiles:
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


def get_all_gs_structure(num_voters=None, num_candidates=None, all_gs_profiles=None):
    if all_gs_profiles is None:
        if num_voters is None and num_candidates is None:
            raise ValueError(
                "You need to provide either number of voters and candidates or a "
                "collection of profiles"
            )
        all_gs_profiles = get_all_group_separable_profiles(num_voters, num_candidates)
    return list(set(gs_structure(p) for p in all_gs_profiles))


def get_all_subsets(num_candidates=None) -> list[set[int]]:
    elements = list(range(num_candidates))
    subsets = []
    for r in range(num_candidates + 1):
        subsets.extend(combinations(elements, r))
    return [set(subset) for subset in subsets]


def hamming(vote_1, vote_2):
    return len(vote_1.symmetric_difference(vote_2))
