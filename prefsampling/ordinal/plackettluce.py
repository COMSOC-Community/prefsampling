import copy

import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def plackett_luce(
        num_voters: int, num_candidates: int, alphas: list[float], seed: int = None
) -> np.ndarray:
    """
    Generates votes according to Plackett-Luce model.

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    alphas : list[float]
        List of model parameters.
    seed : int
        Seed for numpy random number generator.

    Returns
    -------
    np.ndarray
        The votes.

    """

    if len(alphas) != num_candidates:
        raise ValueError(f'Length of alphas should be equal to num_candidates')

    rng = np.random.default_rng(seed)

    alphas = np.array(alphas)

    votes = np.zeros([num_voters, num_candidates])

    for i in range(num_voters):

        items = list(range(num_candidates))
        tmp_alphas = copy.deepcopy(alphas)

        for j in range(num_candidates):
            probabilities = tmp_alphas / sum(tmp_alphas)
            print(items, probabilities)
            chosen = rng.choice(items, p=probabilities)
            votes[i][j] = chosen

            tmp_alphas = np.delete(tmp_alphas, items.index(chosen))
            items.remove(chosen)

    return votes
