import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def single_crossing(
    num_voters: int, num_candidates: int, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes that are single-crossing. See `Elkind, Lackner, Peters (2022)
    <https://arxiv.org/abs/2205.09092>`_ for the definition.

    This sampler works as follows. We generate a random domain of single-crossing, and then randomly
    selects the votes from the domain. The votes in the domain are generated one by one. The first
    vote is always `0 > 1 > 2 > ...`. For vote number `k`, all valid swaps of consecutive candidates
    are considered where a swap is only valid if in vote number `k - 1` the two candidates had not
    already been swapped in previous iterations. One valid swap is selected uniformly at random
    and performed on vote number `k`. One can check that after `m * (m-1) / 2 + 1` votes have been
    generated, no valid swap exists (the final vote being `m > m - 1 > ...`). Once the domain is
    set, we select for each voter one vote from the domain uniformly at random.

    TODO: This is wrong

    This procedure ensures that every set of single-crossing votes for `num_voters` and
    `num_candidates` is equally likely to occur.

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

    domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)
    domain = [list(range(num_candidates))]

    for line in range(1, domain_size):
        all_swap_indices = [
            (j, j + 1)
            for j in range(num_candidates - 1)
            if domain[line - 1][j] < domain[line - 1][j + 1]
        ]
        swap_indices = all_swap_indices[rng.integers(0, len(all_swap_indices))]

        new_line = domain[line - 1].copy()
        new_line[swap_indices[0]] = domain[line - 1][swap_indices[1]]
        new_line[swap_indices[1]] = domain[line - 1][swap_indices[0]]
        domain.append(new_line)

    votes = np.zeros([num_voters, num_candidates], dtype=int)
    for j in range(num_voters):
        r = rng.integers(0, domain_size)
        votes[j, :] = domain[r]

    return votes
