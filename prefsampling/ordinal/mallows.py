"""
Mallows's model is a sampling model parameterised by a central ranking. The probability of
generating a given ranking is then exponential in the distance between the ranking and the central
ranking.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from prefsampling.combinatorics import kendall_tau_distance, all_rankings
from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int
from prefsampling.ordinal import impartial


@validate_num_voters_candidates
def mallows(
    num_voters: int,
    num_candidates: int,
    phi: float,
    normalise_phi: bool = False,
    central_vote: np.ndarray = None,
    impartial_central_vote: bool = False,
    seed: int = None,
) -> list[list[int]]:
    """
    Generates votes according to Mallows' model (`Mallows, 1957
    <https://www.jstor.org/stable/2333244>`_). This model is parameterised by a central vote. The
    probability of generating a given decreases exponentially with the distance between the vote
    and the central vote.

    Specifically, the probability of generating a vote is proportional to `phi ** distance` where
    `phi` is a dispersion coefficient (in [0, 1]) and `distance` is the Kendall-Tau distance between
    the central vote and the vote under consideration. A set of `num_voters` vote is generated
    independently and identically following this process.

    The `phi` coefficient controls the dispersion of the votes: values close to 0 render votes that
    are far away from the central vote unlikely to be generated; and the opposite for values close
    to 1. Depending on the application, it can be advised to normalise the value of `phi`
    (especially when comparing different values for `phi`), see `Boehmer, Faliszewski and Kraiczy
    (2023) <https://proceedings.mlr.press/v202/boehmer23b.html>`_ for more details. Use
    :code:`normalise_phi = True` to do so.

    For an analogous sampler generating approval ballots, see
    :py:func:`~prefsampling.approval.noise.noise`.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        phi : float
            The dispersion coefficient.
        normalise_phi : bool, default: :code:`False`
            Indicates whether phi should be normalised (see `Boehmer, Faliszewski and Kraiczy (2023)
            <https://proceedings.mlr.press/v202/boehmer23b.html>`_)
        central_vote : np.ndarray, default: :code:`np.arrange(num_candidates)`
            The central vote. Ignored if :code:`impartial_central_vote = True`.
        impartial_central_vote: bool, default: :code:`False`
            If true, the central vote is sampled from :py:func:`~prefsampling.ordinal.impartial`.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import mallows

            # Sample from a Mallows' model with 2 voters and 3 candidates, the parameter phi is 0.6
            mallows(2, 3, 0.6)

            # For reproducibility, you can set the seed.
            mallows(2, 3, 1, seed=1002)

            # Parameter phi should be in [0, 1]
            try:
                mallows(2, 3, -0.5)
            except ValueError:
                pass
            try:
                mallows(2, 3, 1.2)
            except ValueError:
                pass

    Validation
    ----------

        The probability distribution derived from Mallows' model is well known.
        Specifically, given :math:`n` agents and :math:`m` candidates, a parameter :math:`\\phi`
        and a central ranking :math:`\\succ_c`, the probability of generating a ranking
        :math:`\\succ` is equal to:

        .. math::

            \\phi^{d(\\succ, \\succ_c)} \\times
            \\frac{1}{\\prod_{j=1}^m \\frac{1 - \\phi^j}{1 - \\phi}}

        where :math:`d(\\succ, \\succ_c)` is the kendall-tau distance between the ranking and the
        central ranking.

        We test that the observed frequencies of rankings aligns with the theoretical probability
        distribution. The fact that the normalisation of phi does not seem to impact the figure
        is due to the small number of candidates that reduces the distance between phi and its
        normalised value.

        .. image:: ../validation_plots/ordinal/mallows_0_1.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Mallows model with phi=0.1

        .. image:: ../validation_plots/ordinal/mallows_0_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Mallows model with phi=0.5

        .. image:: ../validation_plots/ordinal/mallows_0_8.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Mallows model with phi=0.8

        When :code:`phi` is equal to 1, we are supposed to observe a uniform distribution over all
        rankings.

        .. image:: ../validation_plots/ordinal/mallows_1_0.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a Mallows model with phi=1.0

    References
    ----------
        `Non-null ranking models
        <https://www.jstor.org/stable/2333244>`_,
        *Colin Lingwood Mallows*,
        Biometrica, 44:114–130, 1957.

        `Properties of the Mallows model depending on the number of alternatives: A warning for an
        experimentalist.
        <https://proceedings.mlr.press/v202/boehmer23b/boehmer23b.pdf>`_,
        *Niclas Boehmer, Piotr Faliszewski and Sonja Kraiczy*,
        Proceedings of the International Conference on Machine Learning, 2023.
    """
    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0, 1]")
    if normalise_phi:
        phi = phi_from_norm_phi(num_candidates, phi)

    rng = np.random.default_rng(seed)

    if impartial_central_vote:
        central_vote = impartial(1, num_candidates, seed=seed)[0]

    insert_distributions = [
        _insert_prob_distr(i, phi) for i in range(1, num_candidates)
    ]
    votes = []
    for i in range(num_voters):
        vote = _mallows_vote(num_candidates, insert_distributions, rng=rng)
        if central_vote is not None:
            vote = [central_vote[i] for i in vote]
        votes.append(vote)
    return votes


@validate_num_voters_candidates
def norm_mallows(
    num_voters: int,
    num_candidates: int,
    norm_phi: float,
    central_vote: np.ndarray = None,
    impartial_central_vote: bool = False,
    seed: int = None,
) -> list[list[int]]:
    """
    Shortcut for the function :py:func:`~prefsampling.ordinal.mallows` with
    :code:`normalise_phi = True`.
    """
    if norm_phi < 0 or 1 < norm_phi:
        raise ValueError(
            f"Incorrect value of normphi: {norm_phi}. Value should be in [0,1]"
        )

    return mallows(
        num_voters,
        num_candidates,
        norm_phi,
        normalise_phi=True,
        central_vote=central_vote,
        impartial_central_vote=impartial_central_vote,
        seed=seed,
    )


def _insert_prob_distr(position: int, phi: float) -> np.ndarray:
    """
    Computes the insertion probability distribution for a given position and a given dispersion
    coefficient.

    Parameters
    ----------
    position: int
        The position in the ranking
    phi: float
        The dispersion parameter

    Returns
    -------
    np.ndarray
        The probability distribution.

    """
    distribution = np.zeros(position + 1)
    for j in range(position + 1):
        distribution[j] = phi ** (position - j)
    return distribution / distribution.sum()


def _mallows_vote(
    num_candidates: int,
    insert_distributions: list[np.ndarray],
    rng: np.random.Generator,
) -> list[int]:
    """
    Samples a vote according to Mallows' model.

    Parameters
    ----------
    num_candidates: int
        Number of candidates
    insert_distributions: list[np.ndarray]
        A list of np.ndarray representing the insert probability distributions
    rng: np.random.Generator
        The numpy random generator to use for randomness.

    Returns
    -------
    np.ndarray
        The vote.

    """
    vote = [0]
    for j in range(1, num_candidates):
        insert_distribution = insert_distributions[j - 1]
        index = rng.choice(range(len(insert_distribution)), p=insert_distribution)
        vote.insert(index, j)
    return vote


def _calculate_expected_number_swaps(num_candidates: int, phi: float) -> float:
    """
    Computes the expected number of swaps in a vote sampled from Mallows' model.

    Parameters
    ----------
    num_candidates: int
        The number of candidates
    phi: float
        The dispersion coefficient of the Mallows' model

    Returns
    -------
    float
        The expected number of swaps
    """
    res = phi * num_candidates / (1 - phi)
    for j in range(1, num_candidates + 1):
        res += (j * (phi**j)) / ((phi**j) - 1)
    return res


def phi_from_norm_phi(num_candidates: int, norm_phi: float) -> float:
    """
    Computes an approximation of the dispersion coefficient of a Mallows' model based on its
    normalised coefficient (`norm_phi`).

    Parameters
    ----------
    num_candidates: int
        The number of candidates
    norm_phi: float
        The normalised dispersion coefficient of the Mallows' model

    Returns
    -------
    float
        The (non-normalised) dispersion coefficient of the Mallows' model

    """
    if norm_phi == 1:
        return 1
    if norm_phi > 2 or norm_phi < 0:
        raise ValueError(
            f"The value of norm_phi should be between in (0, 2) (it is now {norm_phi})."
        )
    if norm_phi > 1:
        return 2 - norm_phi
    exp_abs = norm_phi * (num_candidates * (num_candidates - 1)) / 4
    low = 0
    high = 1
    while low <= high:
        mid = (high + low) / 2
        cur = _calculate_expected_number_swaps(num_candidates, mid)
        if abs(cur - exp_abs) < 1e-5:
            return mid

        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid
        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    raise ValueError(
        "Something went wrong when computing phi, we should not have ended up here."
    )


def theoretical_distribution(
    num_candidates: int,
    phi: float,
    normalise_phi: bool = False,
    rankings: Iterable[tuple[int]] = None,
) -> dict:
    validate_int(num_candidates, lower_bound=0)
    if rankings is None:
        rankings = all_rankings(num_candidates)
    distribution = {}
    if normalise_phi:
        phi = phi_from_norm_phi(num_candidates, phi)
    central_ranking = tuple(range(num_candidates))
    for ranking in rankings:
        distribution[ranking] = phi ** kendall_tau_distance(central_ranking, ranking)
    normaliser = sum(distribution.values())
    for r in distribution:
        distribution[r] /= normaliser
    return distribution
