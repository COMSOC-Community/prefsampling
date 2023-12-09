
import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def identity(num_voters: int, num_candidates: int, seed: int = None) -> np.ndarray:
    """
    Generates unanimous ordinal votes.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        np.ndarray
            Ordinal votes.
    """

    return np.array([[j for j in range(num_candidates)] for _ in range(num_voters)])
