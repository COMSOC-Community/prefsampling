import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def mallows(
    num_voters: int,
    num_candidates: int,
    phi: float = 0.5,
    weight: float = 0,
    seed: int = None,
) -> np.ndarray:
    """
    Generates votes according to Mallows' model (Mallows, 1957).

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    phi : float
        The dispersion coefficient.
    weight : float

    seed : int
        Seed for numpy random number generator.

    Returns
    -------
    np.ndarray
        The votes.
    """
    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0, 1]")

    rng = np.random.default_rng(seed)

    insert_distributions = [
        _insert_prob_distr(i, phi) for i in range(1, num_candidates)
    ]
    votes = np.zeros((num_voters, num_candidates))
    for i in range(num_voters):
        vote = _mallows_vote(num_candidates, insert_distributions, rng=rng)
        if weight > 0 and rng.random() <= weight:
            np.flip(vote)
        votes[i] = vote
    return votes


@validate_num_voters_candidates
def norm_mallows(
    num_voters: int,
    num_candidates: int,
    norm_phi: float = 0.5,
    weight: float = 0,
    seed: int = None,
) -> np.ndarray:
    """
    Generates votes according to Mallows' normalised model (Boehmer, Faliszewski and Kraiczy 23).

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    norm_phi : float
        The normalised dispersion coefficient.
    weight : float

    seed : int
        Seed for numpy random number generator.

    Returns
    -------
    np.ndarray
        The votes.
    """
    if norm_phi < 0 or 1 < norm_phi:
        raise ValueError(
            f"Incorrect value of normphi: {norm_phi}. Value should be in [0,1]"
        )

    phi = phi_from_norm_phi(num_candidates, norm_phi)
    return mallows(num_voters, num_candidates, phi, weight, seed)


def _insert_prob_distr(position: int, phi: float) -> np.ndarray:
    """
    Computes the insertion probability distribution for a given position and a given dispersion
    coefficient.

    Parameters
    ----------
    position: int
        The position in the ranking
    phi: float
        The dispersion parameter

    Returns
    -------
    np.ndarray
        The probability distribution.

    """
    distribution = np.zeros(position + 1)
    for j in range(position + 1):
        distribution[j] = phi ** (position - j)
    return distribution / distribution.sum()


def _mallows_vote(
    num_candidates: int,
    insert_distributions: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Samples a vote according to Mallows' model.

    Parameters
    ----------
    num_candidates: int
        Number of candidates
    insert_distributions: list[np.ndarray]
        A list of np.ndarray representing the insert probability distributions
    rng: np.random.Generator
        The numpy random generator to use for randomness.

    Returns
    -------
    np.ndarray
        The vote.

    """
    vote = np.zeros(1)
    for j in range(1, num_candidates):
        insert_distribution = insert_distributions[j - 1]
        index = rng.choice(range(len(insert_distribution)), p=insert_distribution)
        vote = np.insert(vote, index, j)
    return vote


def _calculate_expected_number_swaps(num_candidates: int, phi: float) -> float:
    """
    Computes the expected number of swaps in a vote sampled from Mallows' model.

    Parameters
    ----------
    num_candidates: int
        The number of candidates
    phi: float
        The dispersion coefficient of the Mallows' model

    Returns
    -------
    float
        The expected number of swaps
    """
    res = phi * num_candidates / (1 - phi)
    for j in range(1, num_candidates + 1):
        res += (j * (phi**j)) / ((phi**j) - 1)
    return res


def phi_from_norm_phi(num_candidates: int, norm_phi: float) -> float:
    """
    Computes an approximation of the dispersion coefficient of a Mallows' model based on its
    normalised coefficient (`norm_phi`).

    Parameters
    ----------
    num_candidates: int
        The number of candidates
    norm_phi: float
        The normalised dispersion coefficient of the Mallows' model

    Returns
    -------
    float
        The (non-normalised) dispersion coefficient of the Mallows' model

    """
    if norm_phi == 1:
        return 1
    if norm_phi > 2 or norm_phi < 0:
        raise ValueError(
            f"The value of norm_phi should be between in (0, 2) (it is now {norm_phi})."
        )
    if norm_phi > 1:
        return 2 - norm_phi
    exp_abs = norm_phi * (num_candidates * (num_candidates - 1)) / 4
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

    raise ValueError(
        "Something went wrong when computing phi, we should not have ended up here."
    )
