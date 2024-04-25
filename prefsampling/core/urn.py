from __future__ import annotations

from collections.abc import Callable

import numpy as np


def urn_scheme(
    num_samples: int,
    alpha: float,
    base_case_sampler: Callable,
    rng: np.random.Generator,
) -> list:
    """
    Generates votes following a Pólya-Eggenberger urn process. This is the general scheme that is
    used, for instance, in :py:func:`~prefsampling.ordinal.urn`.

    When generating a sample the following happens. With a probability of 1/(urn_size + 1), the
    base case sample is generated (based on :code:`base_case_sampler`). With probability
    `1/urn_size`, an element of the urn is selected uniformly at random. In both cases, the element
    is put back in the urn together with `alpha * num_different_balls` copies of the vote
    (where `num_different_balls` is the number different outcomes of the :code:`base_case_sampler`
    function).

    Parameters
    ----------
        num_samples: int
            The number of samples to select
        alpha: float
            The dispersion coefficient (`alpha * m!` copies of a vote are put back in the urn after
            a draw). Must be non-negative.
        base_case_sampler: Callable
            A function that returns a sample to add in the urn in the base case. It should be a
            function that takes a random number generator as its one and only argument.
        rng : np.random.Generator
            The random number generator used

    Returns
    -------
        list
            A list of samples
    """

    if alpha < 0:
        raise ValueError("Alpha needs to be non-negative for an urn model.")

    balls = []
    urn_size = 1.0
    for i in range(num_samples):
        if rng.uniform(0, urn_size) <= 1.0:
            balls.append(base_case_sampler(rng))
        else:
            balls.append(balls[rng.integers(0, i)])
        urn_size += alpha
    return balls
