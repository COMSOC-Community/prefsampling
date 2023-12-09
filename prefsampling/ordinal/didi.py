import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def generate_didi_votes(
        num_voters: int, num_candidates: int, alphas: list[float] = None, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes from DiDi model.


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
        list[set[int]]
            Approval votes.

    Raises
    ------
        ValueError
            When len(`alphas`) not equal num_candidates.
    """

    if alphas is None:
        alphas = [1/num_candidates for _ in range(num_candidates)]

    if len(alphas) != num_candidates:
        raise ValueError(f"Incorrect length of alphas vector. Should be equal to num_candidates.")

    rng = np.random.default_rng(seed)

    votes = [[0 for _ in range(num_candidates)] for _ in range(num_voters)]

    for j in range(num_voters):
        points = rng.dirichlet(alphas)
        cand = [q for q in range(num_candidates)]
        tmp_candidates = [x for _, x in sorted(zip(points, cand))]
        for k in range(num_candidates):
            votes[j][k] = tmp_candidates[num_candidates - k - 1]

    return np.array(votes)
