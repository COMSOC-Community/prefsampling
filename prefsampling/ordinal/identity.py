"""
Identity samplers are not fascinating per se as all voters have the same preferences. There are
useful tools however, for instance when using them in mixtures.
"""

from __future__ import annotations

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def identity(num_voters: int, num_candidates: int, seed: int = None) -> list[list[int]]:
    """
    Generates unanimous ordinal votes, all votes being `0, 1, 2, ...`.

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
        list[list[int]]
            Ordinal votes.


    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import identity

            # Sample a unanimous profile with 2 voters and 3 candidates
            identity(2, 3)

            # The seed will not change anything here, but you can still set it.
            identity(2, 3, seed=1002)

    Validation
    ----------
        Validation is trivial here, we thus omit it.
    """

    return [list(range(num_candidates)) for _ in range(num_voters)]
