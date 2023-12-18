import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def didi(
    num_voters: int, num_candidates: int, alphas: list[float], seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes from the DiDi (Dirichlet Distribution) model.

    This model is parameterised by a vector `alphas` intuitively indicating a quality for each
    candidate. Moreover, the higher the sum of the `alphas`, the more correlated the votes are
    (the more concentrated the Dirichlet distribution is). To sample a vote, we sample a set of
    points---one per candidate---from a Dirichlet distribution parameterised by `alphas`. The
    vote then corresponds to the candidates ordered by decreasing order of points.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    This model is very similar in spirit to the :py:func:`~prefsampling.ordinal.plackett_luce`
    model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        alphas : list[float]
            List of model params.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.

    Raises
    ------
        ValueError
            When len(`alphas`) not equal num_candidates.
    """
    if len(alphas) != num_candidates:
        raise ValueError(
            "Incorrect length of alphas vector. Should be equal to num_candidates."
        )

    if not all(a >= 0 for a in alphas):
        raise ValueError("The values of the alpha vector should all be positive.")

    rng = np.random.default_rng(seed)

    votes = np.zeros((num_voters, num_candidates), dtype=int)

    for i in range(num_voters):
        points = rng.dirichlet(alphas)
        votes[i] = np.flip(points.argsort())

    return np.array(votes)
