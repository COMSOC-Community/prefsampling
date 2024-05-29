"""
Plackett-Luce models are parameterised by a vector of quality for the candidates. The higher the
quality of a candidate, the higher the change that they show up high in the rankings.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from prefsampling.combinatorics import all_rankings
from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int


@validate_num_voters_candidates
def plackett_luce(
    num_voters: int, num_candidates: int, alphas: list[float], seed: int = None
) -> list[list[int]]:
    """
    Generates ordinal votes according to Plackett-Luce model.

    This model is parameterised by a vector `alphas` intuitively indicating a quality for each
    candidate. A vote is generated in `m` steps (`m` being the number of candidates). The vote is
    filled up from most preferred to least preferred candidate. For the initial draw, the
    probability of selecting a candidate is equal to its quality (normalised). Then, this candidate
    is removed and the quality are re-normalised.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    For a similar model, see the :py:func:`~prefsampling.ordinal.didi.didi` model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        alphas : list[float]
            List of model parameters (quality of the candidates).
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import plackett_luce

            # Sample from a Plackett-Luce model with 2 voters and 3 candidates, the qualities of
            # candidates are 0.5, 0.2, and 0.1.
            plackett_luce(2, 3, (0.5, 0.2, 0.1))

            # For reproducibility, you can set the seed.
            plackett_luce(2, 3, (5, 2, 0.7), seed=1002)

            # Don't forget to provide a quality for all candidates
            try:
                plackett_luce(2, 3, (0.5, 0.2))
            except ValueError:
                pass

            # And all quality scores need to be strictly positive
            try:
                plackett_luce(2, 3, (0.5, 0.2, -0.4))
            except ValueError:
                pass
            try:
                plackett_luce(2, 3, (0.5, 0.2, 0))
            except ValueError:
                pass

    Validation
    ----------

        The probability distribution governing the Plackett-Luce model is well documented.
        Specifically, given :math:`n` agents and :math:`m` candidates and a vector of candidates
        qualities :math:`\\mathbf{\\alpha} = (\\alpha_1, \\ldots, \\alpha_m)`, the probability of
        generating a ranking :math:`a_{i_1} \\succ a_{i_2} \\succ \\cdots \\succ a_{i_m}`
        is equal to:

        .. math::

            \\frac{\\alpha_{i_1}}{1} \\times \\frac{\\alpha_{i_2}}{\\sum_{p > 1}\\alpha_{i_p}}
            \\times \\cdots \\times \\frac{\\alpha_{i_{m-1}}}{\\alpha_{i_{m-1}} + \\alpha_{i_m}}.

        We test that the observed frequencies of rankings aligns with the theoretical probability
        distribution.

        .. image:: ../validation_plots/ordinal/plackett_luce__0_629359640_236676380_031727720_969677880_83727159_.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Plackett-Luce model with alpha=[0.62935964 0.23667638 0.03172772 0.96967788 0.83727159]

        .. image:: ../validation_plots/ordinal/plackett_luce__1_0_0_3_0_3_0_3_0_3_.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Plackett-Luce model with alpha=[1.0, 0.3, 0.3, 0.3, 0.3]

        When all values in :code:`alphas` are equal, we are supposed to observe a uniform
        distribution over all rankings.

        .. image:: ../validation_plots/ordinal/plackett_luce__0_1_0_1_0_1_0_1_0_1_.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Plackett-Luce model with alpha=[0.1, 0.1, 0.1, 0.1, 0.1]

    References
    ----------
        `Individual Choice Behavior: A Theoretical Analysis
        <https://psycnet.apa.org/record/1960-03588-000>`_,
        *R. Duncan Luce*,
        New York: Wiley, 1959.

        `The Analysis of Permutations
        <https://www.jstor.org/stable/2346567>`_,
        *Robert L. Plackett*,
        Applied Statistics 24 (2): 193–202, 1975.

        `Learning Mixtures of Plackett-Luce Models
        <https://proceedings.mlr.press/v48/zhaob16.pdf>`_,
        *Zhibing Zhao, Peter Piech and Lirong Xia*,
        Proceedings of the International Conference on Machine Learning (pp. 2906-2914), 2016.
    """

    if len(alphas) != num_candidates:
        raise ValueError("Length of alphas should be equal to num_candidates")
    if min(alphas) <= 0:
        raise ValueError("The alphas should all be strictly greater than 0.")

    rng = np.random.default_rng(seed)

    alphas = np.array(alphas, dtype=float)

    votes = []

    for i in range(num_voters):
        items = list(range(num_candidates))
        tmp_alphas = alphas.copy()

        vote = []
        for j in range(num_candidates):
            probabilities = tmp_alphas / sum(tmp_alphas)
            chosen = rng.choice(items, p=probabilities)
            vote.append(chosen)

            tmp_alphas = np.delete(tmp_alphas, items.index(chosen))
            items.remove(chosen)
        votes.append(vote)
    return votes


def theoretical_distribution(
    alphas, num_candidates: int = None, rankings: Iterable[tuple[int]] = None
) -> dict:
    if rankings is None:
        if num_candidates is None:
            raise ValueError(
                "If you do not provide the collection of rankings, you need to "
                "provide the number of candidates."
            )
        validate_int(num_candidates, lower_bound=0)
        rankings = all_rankings(num_candidates)
    distribution = {}
    norm_alphas = np.array(alphas, dtype=float) / sum(alphas)
    for ranking in rankings:
        probability = 1
        for j, alt in enumerate(ranking):
            probability *= norm_alphas[alt] / np.take(norm_alphas, ranking[j:]).sum()
        distribution[ranking] = probability
    normaliser = sum(distribution.values())
    for r in distribution:
        distribution[r] /= normaliser
    return distribution
