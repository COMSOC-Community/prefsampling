from __future__ import annotations

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates

from prefsampling.core.urn import urn_scheme


@validate_num_voters_candidates
def urn(
    num_voters: int, num_candidates: int, alpha: float, seed: int = None
) -> np.ndarray:
    """
    Generates votes following the Pólya-Eggenberger urn culture. The process is as follows. The urn
    is initially empty and votes are generated one after the other, in turns. When generating a
    vote, the following happens. With a probability of 1/(urn_size + 1), the vote is selected
    uniformly at random (following an impartial culture). With probability `1/urn_size` a vote
    from the urn is selected uniformly at random. In both cases, the vote is put back in the urn
    together with `alpha * m!` copies of the vote (where `m` is the number of candidates).

    Note that for a given number of voters, votes are not sampled independently.

    Parameters
    ----------
        num_voters: int
            Number of voters
        num_candidates: int
            Number of candidates
        alpha: float
            The dispersion coefficient (`alpha * m!` copies of a vote are put back in the urn after
            a draw). Must be non-negative.
        seed: int, default: :code:`None`
            The seed for the random number generator.

    Returns
    -------
        np.ndarray
            The votes

    Validation
    ----------

        The probability distribution governing an urn model is well documented.

        When :code:`alpha = 1 / m!`, we fall back to the case of the impartial anonymous
        culture. For other values of :code:`alpha`, different probability distributions are
        obtained.

        With the impartial anonymous culture, every multisets of votes--an anonymous profile---are
        equally likely to be generated. Note here that we are discussing anonymous profiles and
        not ranks.

        .. image:: ../validation_plots/ordinal/urn.png
            :width: 600
            :alt: Observed versus theoretical frequencies for an urn model with alpha=0

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import urn

            # Sample from an urn model with 2 voters and 3 candidate. Urn parameter is 0.5.
            urn(2, 3, 0.5)

            # For reproducibility, you can set the seed.
            urn(2, 3, 4, seed=1002)

            # Passing a negative alpha will fail
            try:
                urn(2, 3, -0.5)
            except ValueError:
                pass


    References
    ----------
        `Paradox of Voting under an Urn Model: The Effect of Homogeneity
        <https://www.jstor.org/stable/30024551>`_,
        *Sven Berg*,
        Public Choice, Vol. 47, No. 2 (1985).
    """
    rng = np.random.default_rng(seed)

    votes = urn_scheme(num_voters, alpha, lambda x: x.permutation(num_candidates), rng)
    return np.array(votes, dtype=int)


def theoretical_distribution(num_voters, num_candidates, alpha) -> dict:
    def ascending_factorial(value, length, increment):
        if length == 0:
            return 1
        return (
            value
            + (length - 1)
            * increment
            * math.factorial(sampler_parameters["num_candidates"])
        ) * ascending_factorial(value, length - 1, increment)

    distribution = {}
    for profile in all_profiles():
        counts = {}
        for rank in profile:
            if rank in counts:
                counts[rank] += 1
            else:
                counts[rank] = 1
        probability = math.factorial(
            sampler_parameters["num_voters"]
        ) / ascending_factorial(
            math.factorial(sampler_parameters["num_candidates"]),
            sampler_parameters["num_voters"],
            sampler_parameters["alpha"],
        )
        for c in counts.values():
            probability *= ascending_factorial(
                1, c, sampler_parameters["alpha"]
            ) / math.factorial(c)
        distribution[profile] = probability
    normaliser = sum(distribution.values())
    for r in distribution:
        distribution[r] /= normaliser
    return distribution