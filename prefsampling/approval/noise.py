import math
from enum import Enum

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


class NoiseType(Enum):
    """
    Constants representing the different types of noise that can be applied to the noise sampler.
    """

    HAMMING = 1
    """
    Hamming noise.
    """

    JACCARD = 2
    """
    Jaccard noise.
    """

    ZELINKA = 3
    """
    Zelinka noise.
    """

    BUNKE_SHEARER = 4
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

    A = {i for i in range(k)}
    B = set(range(num_candidates)) - A

    choices = []
    probabilities = []

    # Prepare buckets
    for x in range(len(A) + 1):
        num_options_in = math.comb(len(A), x)
        for y in range(len(B) + 1):
            num_options_out = math.comb(len(B), y)

            if noise_type == NoiseType.HAMMING:
                factor = phi ** (len(A) - x + y)
            elif noise_type == NoiseType.JACCARD:
                factor = phi ** ((len(A) - x + y) / (len(A) + y))
            elif noise_type == NoiseType.ZELINKA:
                factor = phi ** max(len(A) - x, y)
            elif noise_type == NoiseType.BUNKE_SHEARER:
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
        _id = rng.choice(range(len(choices)), 1, p=probabilities)[0]
        x, y = choices[_id]
        vote = set(rng.choice(list(A), x, replace=False))
        vote = vote.union(set(rng.choice(list(B), y, replace=False)))
        votes.append(vote)

    return votes
