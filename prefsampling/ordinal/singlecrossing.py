import numpy as np


def single_crossing(
        num_voters: int,
        num_candidates: int,
        seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes that are single-crossing.

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    seed : int
        Seed for numpy random number generator.

    Returns
    -------
    np.ndarray
        Ordinal votes.
    """
    rng = np.random.default_rng(seed)
    votes = np.zeros([num_voters, num_candidates])

    domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)
    domain = [list(range(num_candidates)) for _ in range(domain_size)]

    for line in range(1, domain_size):
        swap_candidates = [
            (i, i + 1)
            for i in range(num_candidates - 1)
            if domain[line - 1][i] < domain[line - 1][i + 1]
        ]
        swap_indices = swap_candidates[rng.integers(0, len(swap_candidates))]

        domain[line] = domain[line - 1].copy()
        domain[line][swap_indices[0]], domain[line][swap_indices[1]] = (
            domain[line][swap_indices[1]],
            domain[line][swap_indices[0]],
        )

    for j in range(num_voters):
        r = rng.integers(0, domain_size)
        votes[j] = list(domain[r])

    return votes
