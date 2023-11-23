import numpy as np


def mallows(
    num_voters: int = None,
    num_candidates: int = None,
    phi: float = 0.5,
    weight: float = 0,
    seed: int = None,
):
    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0,1]")

    rng = np.random.default_rng(seed)

    insertion_probabilites_list = []
    for i in range(1, num_candidates):
        insertion_probabilites_list.append(_compute_insertion_probas(i, phi))
    votes = []
    for i in range(num_voters):
        vote = _mallows_vote(num_candidates, insertion_probabilites_list, rng=rng)
        if weight > 0:
            probability = rng.random()
            if probability <= weight:
                vote.reverse()
        votes += [vote]
    return votes


def norm_mallows(
    num_voters: int = None,
    num_candidates: int = None,
    normphi: float = 0.5,
    weight: float = 0,
    seed: int = None,
):
    if normphi < 0 or 1 < normphi:
        raise ValueError(
            f"Incorrect value of normphi: {normphi}. Value should be in [0,1]"
        )

    phi = _phi_from_normphi(num_candidates, normphi)
    return mallows(num_voters, num_candidates, phi, weight, seed)


def _compute_insertion_probas(i, phi):
    probas = (i + 1) * [0]
    for j in range(i + 1):
        probas[j] = pow(phi, (i + 1) - (j + 1))
    return probas


def _mallows_vote(m, insertion_probabilites_list, rng):
    vote = [0]
    for i in range(1, m):
        index = _weighted_choice(insertion_probabilites_list[i - 1], rng=rng)
        vote.insert(index, i)
    return vote


def _weighted_choice(choices, rng):
    total = 0
    for w in choices:
        total = total + w
    r = rng.uniform(0, total)
    upto = 0.0
    for i, w in enumerate(choices):
        if upto + w >= r:
            return i
        upto = upto + w
    assert False, "Shouldn't process_id get here"


def _calculate_expected_number_swaps(num_candidates, phi):
    """
    Given the number m of candidates and a phi\in [0,1] function computes
    the expected number of swaps in a vote sampled from Mallows culture_id
    """
    res = phi * num_candidates / (1 - phi)
    for j in range(1, num_candidates + 1):
        res = res + (j * (phi**j)) / ((phi**j) - 1)
    return res


def _phi_from_normphi(num_candidates=None, normphi=None):
    """
    Given the number m of candidates and a absolute number of expected swaps exp_abs, this function
    returns a value of phi such that in a vote sampled from Mallows culture_id with this parameter
    the expected number of swaps is exp_abs
    """
    if normphi is None:
        raise ValueError("normphi is not defined")
    if normphi == 1:
        return 1
    if normphi > 2 or normphi < 0:
        raise ValueError("Incorrect normphi value")
    if normphi > 1:
        return 2 - normphi
    exp_abs = normphi * (num_candidates * (num_candidates - 1)) / 4
    low = 0
    high = 1
    while low <= high:
        mid = (high + low) / 2
        cur = _calculate_expected_number_swaps(num_candidates, mid)
        if abs(cur - exp_abs) < 1e-5:
            return mid
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    # If we reach here, then the element was not present
    return -1
