import numpy as np

from prefsampling.decorators import validate_num_agents_candidates


@validate_num_agents_candidates
def urn(
    num_voters: int, num_candidates: int, alpha: float = 0.1, seed: int = None
) -> np.ndarray:
    """
    Generates votes following the Pólya-Eggenberger urn culture. The process is a follows. The urn is initially empty
    and votes are generated one after the other, in turns. When generating a vote, the following happens. With
    probability `1/(urn_size + 1)` the vote is simply selected uniformly at random (following an impartial culture).
    With probability `1/urn_size` a vote from the urn is selected uniformly at random. In both cases, the vote is
    put back in the urn together with `alpha` copies.

    Parameters
    ----------
    num_voters: int
        The number of voters
    num_candidates: int
        The number of candidates
    alpha: float
        The coefficient alpha
    seed: int
        The seed for the random number generator.

    Returns
    -------
        np.ndarray
            The votes
    """

    if alpha < 0:
        raise ValueError("Alpha need to be positive for an urn model.")

    rng = np.random.default_rng(seed)
    votes = np.zeros((num_voters, num_candidates), dtype=int)
    urn_size = 1.0
    for i in range(num_voters):
        rho = rng.uniform(0, urn_size)
        if rho <= 1.0:
            votes[i] = rng.permutation(num_candidates)
        else:
            votes[i] = votes[rng.integers(0, i)]
        urn_size += alpha

    return votes
