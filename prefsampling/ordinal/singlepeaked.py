"""
Single-peaked preferences capture the idea that there exists a societal axis on which the candidates
can be placed in such a way that the preferences of the voters are decreasing along both sides of
the axis from their most preferred candidate.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from prefsampling.combinatorics import (
    all_single_peaked_rankings,
    all_single_peaked_circle_rankings,
)
from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int


@validate_num_voters_candidates
def single_peaked_conitzer(
    num_voters: int, num_candidates: int, axis: list[int] = None, seed: int = None
) -> list[list[int]]:
    """
    Generates ordinal votes that are single-peaked following the distribution defined by
    `Conitzer (2009) <https://arxiv.org/abs/1401.3449>`_. The preferences generated are
    single-peaked with respect to the axis `0, 1, 2, ...`. Votes are generated uniformly at random
    as follows. The most preferred candidate (the peak) is selected uniformly at random. Then,
    either the candidate on the left, or the one on the right of the peak comes second in the
    ordering. Each case occurs with probability 0.5. The method is then iterated for the next left
    and right candidates (only one of them being different from before).

    This method ensures that the probability for a given candidate to be the peak is uniform
    (as opposed to the method :py:func:`~prefsampling.ordinal.single_peaked_walsh`). The
    probability for a single-peaked rank to be generated is equal to
    `1/m * (1/2)**dist_peak_to_end` where `m` is the number of candidates and `dist_peak_to_end`
    is the minimum distance from to peak to an end of the axis (i.e., candidates `0` or `m - 1`).

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        axis : list[int], default: :code:`None`
            The societal axis
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import single_peaked_conitzer

            # Sample a single-peaked profile via Conitzer's method with 2 voters and 3 candidates.
            single_peaked_conitzer(2, 3)

            # You can set the societal axis
            single_peaked_conitzer(2, 3, axis=[0, 2, 1])

            # For reproducibility, you can set the seed.
            single_peaked_conitzer(2, 3, seed=1002)

    Validation
    ----------
        Following the method proposed by Contizer, the probability of observing a given
        single-peaked ranking (for a fixed axis) with :math:`m` candidates is equal to:

        .. math::

            \\frac{1}{m} \\times \\left(\\frac{1}{2}\\right)^{\\text{dist\\_peak\\_to\\_end}}

        where :math:`\\text{dist\\_peak\\_to\\_end}` is the minimum distance from the top candidate
        to one of the end of the axis (i.e., candidates `0` or `m - 1`).

        .. image:: ../validation_plots/ordinal/sp_conitzer_4.png
            :width: 800
            :alt: Observed versus theoretical frequencies for Conitzer's single-peaked model with m=4

        .. image:: ../validation_plots/ordinal/sp_conitzer_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for Conitzer's single-peaked model with m=5

        .. image:: ../validation_plots/ordinal/sp_conitzer_6.png
            :width: 800
            :alt: Observed versus theoretical frequencies for Conitzer's single-peaked model with m=6


    References
    ----------
        `Eliciting single-peaked preferences using comparison queries
        <https://arxiv.org/abs/1401.3449>`_,
        *Vincent Conitzer*,
        Journal of Artificial Intelligence Research, 35:161–191, 2009.
    """
    if axis is not None:
        if len(axis) != num_candidates or min(axis) < 0 or max(axis) >= num_candidates:
            raise ValueError("The axis should have exactly the same length as the number of "
                             "candidates, they should be named for 0 to num_candidates -  1.")

    rng = np.random.default_rng(seed)
    votes = []
    for _ in range(num_voters):
        peak = rng.choice(range(num_candidates))
        vote = [peak]
        left = peak - 1
        right = peak + 1
        for _ in range(1, num_candidates):
            # If we are stuck on the left or on the right, we fill in the vote
            if left < 0:
                vote += range(right, num_candidates)
                break
            if right >= num_candidates:
                vote += range(left, -1, -1)
                break
            if rng.random() < 0.5:
                vote.append(right)
                right += 1
            else:
                vote.append(left)
                left -= 1
        if axis is not None:
            vote = [axis[c] for c in vote]
        votes.append(vote)

    return votes


def conitzer_theoretical_distribution(
    num_candidates: int, sp_rankings: Iterable[tuple[int]] = None
) -> dict:
    validate_int(num_candidates, lower_bound=1)
    if sp_rankings is None:
        sp_rankings = all_single_peaked_rankings(num_candidates)
    distribution = {}
    for ranking in sp_rankings:
        probability = 1 / num_candidates
        for alt in ranking:
            if alt == 0 or alt == num_candidates - 1:
                break
            probability *= 1 / 2
        distribution[ranking] = probability
    return distribution


@validate_num_voters_candidates
def single_peaked_circle(
    num_voters: int, num_candidates: int, axis: list[int] = None, seed: int = None
) -> list[list[int]]:
    """
    Generates ordinal votes that are single-peaked on a circle following a distribution inspired
    from the one by Conitzer (2009) for single-peakedness on a line (see
    :py:func:`~prefsampling.ordinal.single_peaked_conitzer`). This method starts by
    determining the most preferred candidate (the peak). This is done with uniform probability
    over the candidates. Then, subsequent positions in the ordering are filled by taking either the
    next available candidate on the left or on the right, both cases occuring with probability 0.5.
    Left and right are defined here in terms of the circular axis: `0, 1, ..., m, 1` (can be
    changed by using the :code:`axis` parameter).

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        axis : list[int], default: :code:`None`
            The societal axis
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import single_peaked_circle

            # Sample a single-peaked on a circle profile with 2 voters and 3 candidates.
            single_peaked_circle(2, 3)

            # You can set the societal axis
            single_peaked_circle(2, 3, axis=[0, 2, 1])

            # For reproducibility, you can set the seed.
            single_peaked_circle(2, 3, seed=1002)

    Validation
    ----------
        The sampler for single-peaked on a circle is such that all single-peaked on a circle
        rankings are equally likely to be generated.

        .. image:: ../validation_plots/ordinal/sp_circle_4.png
            :width: 800
            :alt: Observed versus theoretical frequencies for single-peaked on a circle model with m=4

        .. image:: ../validation_plots/ordinal/sp_circle_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for single-peaked on a circle model with m=5

        .. image:: ../validation_plots/ordinal/sp_circle_6.png
            :width: 800
            :alt: Observed versus theoretical frequencies for single-peaked on a circle model with m=6


    References
    ----------
        `Preferences Single-Peaked on a Circle
        <https://www.jair.org/index.php/jair/article/view/11732>`_,
        *Dominik Peters and Martin Lackner*,
        Journal of Artificial Intelligence Research, 68:463–502, 2020.
    """
    if axis is not None:
        if len(axis) != num_candidates or min(axis) < 0 or max(axis) >= num_candidates:
            raise ValueError("The axis should have exactly the same length as the number of "
                             "candidates, they should be named for 0 to num_candidates -  1.")

    rng = np.random.default_rng(seed)
    votes = []
    for _ in range(num_voters):
        vote = [rng.choice(range(num_candidates))]
        left = vote[0] - 1
        left = np.mod(left, num_candidates)
        right = vote[0] + 1
        right = np.mod(right, num_candidates)
        for _ in range(1, num_candidates):
            if rng.random() < 0.5:
                vote.append(left)
                left -= 1
                left = np.mod(left, num_candidates)
            else:
                vote.append(right)
                right += 1
                right = np.mod(right, num_candidates)
        if axis is not None:
            vote = [axis[c] for c in vote]
        votes.append(vote)
    return votes


def circle_theoretical_distribution(
    num_candidates: int = None, sp_circ_rankings: Iterable[tuple[int]] = None
) -> dict:
    if sp_circ_rankings is None:
        if num_candidates is None:
            raise ValueError(
                "If you do not provide the collection of single-peaked on a circle "
                "rankings, you need to provide the number of candidates."
            )
        validate_int(num_candidates, lower_bound=1)
        sp_circ_rankings = all_single_peaked_circle_rankings(num_candidates)
    return {r: 1 / len(sp_circ_rankings) for r in sp_circ_rankings}


@validate_num_voters_candidates
def single_peaked_walsh(
    num_voters: int, num_candidates: int, axis: list[int] = None, seed: int = None
) -> list[list[int]]:
    """
    Generates ordinal votes that are single-peaked following the process described by
    `Walsh (2015) <https://arxiv.org/abs/1503.02766>`_. The votes are generated from least preferred
    to most preferred candidates. A given position in the ordering is filled by selecting, with
    uniform probability, either the leftmost or the rightmost candidates that have not yet been
    positioned in the vote (left and right being defined by the axis `0, 1, 2, ...`).

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        axis : list[int], default: :code:`None`
            The societal axis
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import single_peaked_walsh

            # Sample a single-peaked profile via Walsh's method with 2 voters and 3 candidates.
            single_peaked_walsh(2, 3)

            # You can set the societal axis
            single_peaked_walsh(2, 3, axis=[0, 2, 1])

            # For reproducibility, you can set the seed.
            single_peaked_walsh(2, 3, seed=1002)

    Validation
    ----------
        The method proposed by Walsh ensures that for a given axis, all single-peaked rankings
        are equally likely to be generated.

        .. image:: ../validation_plots/ordinal/sp_walsh_4.png
            :width: 800
            :alt: Observed versus theoretical frequencies for Walsh's single-peaked model with m=4

        .. image:: ../validation_plots/ordinal/sp_walsh_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for Walsh's single-peaked model with m=5

        .. image:: ../validation_plots/ordinal/sp_walsh_6.png
            :width: 800
            :alt: Observed versus theoretical frequencies for Walsh's single-peaked model with m=6


    References
    ----------
        `Generating Single Peaked Votes
        <https://arxiv.org/abs/1503.02766>`_,
        *Toby Walsh*,
        ArXiV preprint, 2015.

    """
    if axis is not None:
        if len(axis) != num_candidates or min(axis) < 0 or max(axis) >= num_candidates:
            raise ValueError("The axis should have exactly the same length as the number of "
                             "candidates, they should be named for 0 to num_candidates -  1.")

    rng = np.random.default_rng(seed)
    votes = []

    for _ in range(num_voters):
        left_most = 0
        right_most = num_candidates - 1
        vote = []
        for _ in range(num_candidates):
            if rng.random() < 0.5:
                vote.append(left_most)
                left_most += 1
            else:
                vote.append(right_most)
                right_most -= 1
        if axis is not None:
            vote = [axis[c] for c in vote]
        votes.append(list(reversed(vote)))

    return votes


def walsh_theoretical_distribution(
    num_candidates: int = None, sp_rankings: Iterable[tuple[int]] = None
) -> dict:
    if sp_rankings is None:
        if num_candidates is None:
            raise ValueError(
                "If you do not provide the collection of single-peaked rankings, you "
                "need to provide the number of candidates."
            )
        validate_int(num_candidates, lower_bound=1)
        sp_rankings = all_single_peaked_rankings(num_candidates)
    return {r: 1 / len(sp_rankings) for r in sp_rankings}
