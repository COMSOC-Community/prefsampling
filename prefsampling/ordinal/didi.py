"""
The Dirichlet distribution model is a model parameterised by a vector of candidate quality.
A quality score is associated to each candidate. When sampling a ranking, the quality scores
are used to sample a number of points for each candidate (using a Dirichlet distribution).
The ranking corresponds then to the candidates ordered by number of points.
"""

from __future__ import annotations

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def didi(
    num_voters: int, num_candidates: int, alphas: list[float], seed: int = None
) -> list[list[int]]:
    """
    Generates ordinal votes from the DiDi (Dirichlet Distribution) model.

    This model is parameterised by a vector `alphas` intuitively indicating a quality for each
    candidate. Moreover, the higher the sum of the `alphas`, the more correlated the votes are
    (the more concentrated the Dirichlet distribution is). To sample a vote, we sample a set of
    points---one per candidate---from a Dirichlet distribution parameterised by `alphas`. The
    vote then corresponds to the candidates ordered by decreasing order of points.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    This model is very similar in spirit to the
    :py:func:`~prefsampling.ordinal.plackettluce.plackett_luce` model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        alphas : list[float]
            List of model params, one value per candidate.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import didi

            # Sample from a DiDi model with 2 voters and 3 candidates, the qualities of
            # candidates are 0.5, 0.2, and 0.1.
            didi(2, 3, (0.5, 0.2, 0.1))

            # For reproducibility, you can set the seed.
            didi(2, 3, (5, 2, 0.1), seed=1002)

            # Don't forget to provide a quality for all candidates
            try:
                didi(2, 3, (0.5, 0.2))
            except ValueError:
                pass

            # And all quality scores need to be strictly positive
            try:
                didi(2, 3, (0.5, 0.2, -0.4))
            except ValueError:
                pass
            try:
                didi(2, 3, (0.5, 0.2, 0))
            except ValueError:
                pass

    Validation
    ----------

        The probability distribution guiding the DiDi model is not known in general. Since it
        depends on the order of the values in a Dirichlet sample, the general computation is
        involved. Still, we can check some special cases.

        First, when all qualities are the same, we are supposed to obtain a uniform distribution
        over all rankings.

        .. image:: ../validation_plots/ordinal/didi__0_1_0_1_0_1_0_1_0_1_.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a DiDi model with alpha=[0.1, 0.1, 0.1, 0.1, 0.1]

        Second, in the special case of 2 candidates, we can easily compute an expression for the
        probability distribution of the model. Assume we have two candidates with quality
        :math:`\\alpha_0` and :math:`\\alpha_1`. Then, the probability of observing the ranking
        :math:`0 \\succ 1` is that of the probability to sample two values :math:`x_0`, :math:`x_1`
        from a Dirichlet distribution with parameters :math:`\\alpha_0` and :math:`\\alpha_1` such
        that :math:`x_0 > x_1`. We have thus:

        .. math::

            \\mathbb{P}(x_0 > x_1) = \\mathbb{P}(x_0 > 0.5) = \\int_{0.5}^1 x_0^{\\alpha_0 - 1}
            \\times (1 - x_0)^{\\alpha_1 - 1} dx_0.

        We can compute an approximate value for of this integral using scipy.

        .. image:: ../validation_plots/ordinal/didi__1_0_0_3_.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a DiDi model with alpha=[0.1, 0.1]

        .. image:: ../validation_plots/ordinal/didi__0_1_0_1_.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a DiDi model with alpha=[1, 0.3]

        In the general case, we obtain the following frequencies.

        .. image:: ../validation_plots/ordinal/didi__0_2_0_5_0_3_0_7_0_2_.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a DiDi model with alpha=[0.2, 0.5, 0.3, 0.7, 0.2]

        .. image:: ../validation_plots/ordinal/didi__1_0_0_3_0_3_0_3_0_3_.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a DiDi model with alpha=[1, 0.3, 0.3, 0.3, 0.3]

    References
    ----------

        The DiDi model has not been references in any publications. Stanisław Szufa introduced out
        of curiosity.

        See the `wikipedia page <https://en.wikipedia.org/wiki/Dirichlet_distribution>`_ of the
        Dirichlet distribution for more details.
    """
    if len(alphas) != num_candidates:
        raise ValueError(
            "Incorrect length of alphas vector. Should be equal to num_candidates."
        )

    if not all(a > 0 for a in alphas):
        raise ValueError(
            "The values of the alpha vector should all be strictly positive."
        )

    rng = np.random.default_rng(seed)

    votes = []

    for i in range(num_voters):
        points = rng.dirichlet(alphas)
        votes.append(list(reversed(points.argsort())))

    return votes
