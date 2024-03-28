from __future__ import annotations

import math
from enum import Enum

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates
from prefsampling.utils import comb


class NoiseType(Enum):
    """
    Constants representing the different types of noise that can be applied to the noise sampler.
    """

    HAMMING = "Hamming noise"
    """
    Hamming noise.
    """

    JACCARD = "Jaccard noise"
    """
    Jaccard noise.
    """

    ZELINKA = "Zelinka noise"
    """
    Zelinka noise.
    """

    BUNKE_SHEARER = "Bunke-Shearer noise"
    """
    Bunke-Shearer noise.
    """


@validate_num_voters_candidates
def noise(
    num_voters: int,
    num_candidates: int,
    p: float,
    phi: float,
    noise_type: NoiseType = NoiseType.HAMMING,
    seed: int = None,
) -> list[set]:
    """
    Generates approval votes under the noise model.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        phi : float
            Noise model parameter, denoting the noise.
        p : float
            Noise model parameter, denoting the length of central vote.
        noise_type : NoiseType, default: :py:const:`~prefsampling.approval.NoiseType.HAMMING`
            Type of noise.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set]
            Approval votes.

    Raises
    ------
        ValueError
            When `phi` not in [0,1] interval.
            When `p` not in [0,1] interval.
    """

    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0,1]")

    if p < 0 or 1 < p:
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0,1]")

    rng = np.random.default_rng(seed)

    k = math.floor(p * num_candidates)

    A = list(range(k))
    B = list(range(k + 1, num_candidates))

    choices = []
    probabilities = []

    if isinstance(noise_type, Enum):
        noise_type = NoiseType(noise_type.value)
    else:
        noise_type = NoiseType(noise_type)
    # Prepare buckets
    for x in range(len(A) + 1):
        num_options_in = comb(len(A), x)
        for y in range(len(B) + 1):
            num_options_out = comb(len(B), y)

            if noise_type == NoiseType.HAMMING:
                factor = phi ** (len(A) - x + y)
            elif noise_type == NoiseType.JACCARD:
                if len(A) + y == 0:
                    factor = int(phi == 0)
                else:
                    factor = phi ** ((len(A) - x + y) / (len(A) + y))
            elif noise_type == NoiseType.ZELINKA:
                factor = phi ** max(len(A) - x, y)
            elif noise_type == NoiseType.BUNKE_SHEARER:
                if max(len(A), x + y) == 0:
                    factor = int(phi == 0)
                else:
                    factor = phi ** (max(len(A) - x, y) / max(len(A), x + y))
            else:
                raise ValueError(
                    "The `noise_type` argument needs to be one of the constant defined in the "
                    "approval.NoiseType enumeration. Choices are: "
                    + ", ".join(str(s) for s in NoiseType)
                )

            num_options = num_options_in * num_options_out * factor

            choices.append((x, y))
            probabilities.append(num_options)
    denominator = sum(probabilities)
    probabilities = [p / denominator for p in probabilities]

    # Sample Votes
    votes = []
    for _ in range(num_voters):
        x, y = rng.choice(choices, p=probabilities)
        vote = set(rng.choice(A, x, replace=False))
        vote = vote.union(set(rng.choice(list(B), y, replace=False)))
        votes.append(vote)

    return votes
