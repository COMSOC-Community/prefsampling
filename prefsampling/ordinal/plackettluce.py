import copy

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def plackett_luce(
    num_voters: int, num_candidates: int, alphas: list[float], seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes according to Plackett-Luce model.

    This model is parameterised by a vector `alphas` intuitively indicating a quality for each
    candidate. A vote is generated in `m` steps (`m` being the number of candidates). The vote is
    filled up from most preferred to least preferred candidate. For the initial draw, the
    probability of selecting a candidate is equal to its quality (normalised). Then, this candidate
    is removed and the quality are re-normalised.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    For a similar model, see the :py:func:`~prefsampling.ordinal.didi` model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        alphas : list[float]
            List of model parameters (quality of the candidates).
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.

    """

    if len(alphas) != num_candidates:
        raise ValueError("Length of alphas should be equal to num_candidates")
    if min(alphas) <= 0:
        raise ValueError("The alphas should all be strictly greater than 0.")

    rng = np.random.default_rng(seed)

    alphas = np.array(alphas)

    votes = np.zeros((num_voters, num_candidates), dtype=int)

    for i in range(num_voters):
        items = list(range(num_candidates))
        tmp_alphas = alphas.copy()

        for j in range(num_candidates):
            probabilities = tmp_alphas / sum(tmp_alphas)
            chosen = rng.choice(items, p=probabilities)
            votes[i, j] = chosen

            tmp_alphas = np.delete(tmp_alphas, items.index(chosen))
            items.remove(chosen)

    return votes
