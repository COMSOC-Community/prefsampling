import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def mallows(
    num_voters: int,
    num_candidates: int,
    phi: float,
    normalise_phi: bool = False,
    central_vote: np.ndarray = None,
    seed: int = None,
) -> np.ndarray:
    """
    Generates votes according to Mallows' model (`Mallows, 1957
    <https://www.jstor.org/stable/2333244>`_). This model is parameterised by a central vote. The
    probability of generating a given decreases exponentially with the distance between the vote
    and the central vote.

    Specifically, the probability of generating a vote is proportional to `phi ** distance` where
    `phi` is a dispersion coefficient (in [0, 1]) and `distance` is the Kendall-Tau distance between
    the central vote and the vote under consideration. A set of `num_voters` vote is generated
    independently and identically following this process.

    The `phi` coefficient controls the dispersion of the votes: values close to 0 render votes that
    are far away from the central vote unlikely to be generated; and the opposite for values close
    to 1. Depending on the application, it can be advised to normalise the value of `phi`
    (especially when comparing different values for `phi`), see `Boehmer, Faliszewski and Kraiczy
    (2023) <https://proceedings.mlr.press/v202/boehmer23b.html>`_ for more details. Use
    :code:`normalise_phi = True` to do so.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        phi : float
            The dispersion coefficient.
        normalise_phi : bool, optional
            Indicates whether phi should be normalised (see `Boehmer, Faliszewski and Kraiczy (2023)
            <https://proceedings.mlr.press/v202/boehmer23b.html>`_)
        central_vote : np.ndarray, default: :code:`np.arrange(num_candidates)`
            The central vote.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """
    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0, 1]")
    if normalise_phi:
        phi = phi_from_norm_phi(num_candidates, phi)

    rng = np.random.default_rng(seed)

    insert_distributions = [
        _insert_prob_distr(i, phi) for i in range(1, num_candidates)
    ]
    votes = np.zeros((num_voters, num_candidates), dtype=int)
    for i in range(num_voters):
        vote = _mallows_vote(num_candidates, insert_distributions, rng=rng)
        if central_vote is not None:
            vote = tuple(central_vote[i] for i in vote)
        votes[i, :] = vote
    return votes


@validate_num_voters_candidates
def norm_mallows(
    num_voters: int,
    num_candidates: int,
    norm_phi: float,
    central_vote: np.ndarray = None,
    seed: int = None,
) -> np.ndarray:
    """
    Shortcut for the function :py:func:`~prefsampling.ordinal.mallows` with
    :code:`normalise_phi = True`.
    """
    if norm_phi < 0 or 1 < norm_phi:
        raise ValueError(
            f"Incorrect value of normphi: {norm_phi}. Value should be in [0,1]"
        )

    return mallows(
        num_voters,
        num_candidates,
        norm_phi,
        normalise_phi=True,
        seed=seed,
        central_vote=central_vote,
    )


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
    vote = np.zeros(1, dtype=int)
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
