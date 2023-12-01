from copy import deepcopy
from itertools import permutations, combinations_with_replacement


def get_all_ranks(num_candidates: int) -> list[tuple[int]]:
    return [tuple(rank) for rank in permutations(range(num_candidates))]


def get_all_anonymous_profiles(num_voters: int, num_candidates: int) -> list[tuple[tuple[int]]]:
    return list(combinations_with_replacement(get_all_ranks(num_candidates), num_voters))


def get_all_profiles(num_voters: int, num_candidates: int) -> list[tuple[tuple[int]]]:
    res = set()
    for prof in get_all_anonymous_profiles(num_voters, num_candidates):
        for p in permutations(prof):
            res.add(tuple(p))
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


def is_single_crossing(profile):
    for j, cand1 in enumerate(profile[0]):
        for cand2 in profile[0][j + 1:]:
            cand1_over_cand2 = True
            for vote in profile:
                if vote.index(cand1) < vote.index(cand2) and not cand1_over_cand2:
                    return False
                elif vote.index(cand1) > vote.index(cand2):
                    cand1_over_cand2 = False
    return True


def get_all_single_crossing_profiles(num_voters: int, num_candidates: int, all_profiles=None):
    if all_profiles is None:
        all_profiles = get_all_profiles(num_voters, num_candidates)
    res = []
    for profile in all_profiles:
        for perm in permutations(profile):
            if is_single_crossing(perm):
                res.append(profile)
                break
    return res
