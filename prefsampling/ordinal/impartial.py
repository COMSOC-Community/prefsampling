"""
Impartial cultures are statistical cultures in which all outcomes are equally likely to be
generated.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Collection

import numpy as np

from prefsampling.combinatorics import all_rankings, all_anonymous_profiles
from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int
from prefsampling.ordinal.urn import urn


@validate_num_voters_candidates
def impartial(
    num_voters: int, num_candidates: int, seed: int = None
) -> list[list[int]]:
    """
    Generates ordinal votes from impartial culture.

    In an impartial culture, all votes are equally likely to occur. In this function, each vote is
    generated by getting a random permutation of the candidates from the random number generator.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

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

            from prefsampling.ordinal import impartial

            # Sample from an impartial culture with 2 voters and 3 candidates
            impartial(2, 3)

            # For reproducibility, you can set the seed.
            impartial(2, 3, seed=1002)

    Validation
    ----------

        Under the impartial culture, all rankings are supposed to be equally likely to be generated.

        .. image:: ../validation_plots/ordinal/impartial_4.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial culture with m=4

        .. image:: ../validation_plots/ordinal/impartial_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial culture with m=5

        .. image:: ../validation_plots/ordinal/impartial_6.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial culture with m=6

    References
    ----------
        `Les théories de l’intérêt général et le problème logique de l’agrégation
        <https://www.persee.fr/doc/ecoap_0013-0494_1952_num_5_4_3831>`_,
        *Georges-Théodule Guilbaud*,
        Economie Appliquée, 5:501–584, 1952.
    """
    rng = np.random.default_rng(seed)
    votes = []
    for i in range(num_voters):
        votes.append(list(rng.permutation(num_candidates)))
    return votes


def impartial_theoretical_distribution(
    num_candidates: int = None, rankings: Iterable[tuple[int]] = None
) -> dict:
    if rankings is None:
        if num_candidates is None:
            raise ValueError(
                "If you do not provide the collection of rankings, you need to "
                "provide the number of candidates."
            )
        validate_int(num_candidates, lower_bound=1)
        rankings = all_rankings(num_candidates)
    return {r: 1 / len(rankings) for r in rankings}


@validate_num_voters_candidates
def impartial_anonymous(
    num_voters: int, num_candidates: int, seed: int = None
) -> list[list[int]]:
    """
    Generates ordinal votes from impartial anonymous culture.

    In an impartial anonymous culture, every multi-set of votes is equally likely to occur. For
    instance with 3 voters and 2 candidates, the probability of observing `a > b, a > b, a > b`
    is 1/4. This probability was 1/8 according to the impartial (but not-anonymous) culture.

    Votes are generated by sampling from an urn model (using :py:func:`~prefsampling.ordinal.urn`)
    with parameter `alpha = 1` (see Lepelley, Valognes 2003).

    Note that votes are not generated independently here.

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

            from prefsampling.ordinal import impartial_anonymous

            # Sample from an impartial culture with 2 voters and 3 candidates
            impartial_anonymous(2, 3)

            # For reproducibility, you can set the seed.
            impartial_anonymous(2, 3, seed=1002)

    Validation
    ----------

        Under the impartial anonymous culture, all anonymous profiles are supposed to be equally
        likely to be generated.

        .. image:: ../validation_plots/ordinal/impartial_anonymous_2.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial anonymous culture with n=2

        .. image:: ../validation_plots/ordinal/impartial_anonymous_3.png
            :width: 800
            :alt: Observed versus theoretical frequencies for an impartial anonymous culture with n=3

    References
    ----------
        ` Voter antagonism and the paradox of voting
        <https://www.jstor.org/stable/1914217>`_,
        *Kiyoshi Kuga and Hiroaki Nagatani*,
        Econometrica, 42(6):1045–1067, 1974.

        `Condorcet paradox and anonymous preference profiles
        <https://www.jstor.org/stable/30022874>`_,
        *William V. Gehrlein and Peter C. Fishburn*,
        Public Choice, 26:1–18, 1978.
    """
    return urn(
        num_voters, num_candidates, alpha=1 / math.factorial(num_candidates), seed=seed
    )


def impartial_anonymous_theoretical_distribution(
    num_voters: int = None,
    num_candidates: int = None,
    anonymous_profiles: Iterable[Collection[tuple[int]]] = None,
) -> dict:
    if anonymous_profiles is None:
        if num_candidates is None:
            raise ValueError(
                "If you do not provide the collection of anonymous profiles, "
                "you need to provide the number of candidates."
            )
        if num_voters is None:
            raise ValueError(
                "If you do not provide the collection of anonymous profiles, "
                "you need to provide the number of voters."
            )
        validate_int(num_candidates, lower_bound=1)
        validate_int(num_voters, lower_bound=1)
        anonymous_profiles = all_anonymous_profiles(num_voters, num_candidates)
    return {r: 1 / len(anonymous_profiles) for r in anonymous_profiles}


@validate_num_voters_candidates
def stratification(
    num_voters: int, num_candidates: int, weight: float, seed: int = None
) -> list[list[int]]:
    """
    Generates ordinal votes from stratification model. In the stratification model, candidates are
    split into two classes. Every voters ranks the candidates of the first class above the
    candidates of the second class. Within a class, the ranking is selected uniformly at random.

    The :code:`weight` parameter is used to define the relative size of the first class.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        weight : float
            Size of the upper class.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[list[int]]
            Ordinal votes.

    Examples
    --------

        .. testcode::

            from prefsampling.ordinal import stratification

            # Sample from a stratification culture with 2 voters and 3 candidates
            stratification(2, 3, 0.5)

            # For reproducibility, you can set the seed.
            stratification(2, 3, 0.2, seed=1002)

            # Parameter weight should be in [0, 1]
            try:
                stratification(2, 3, -0.5)
            except ValueError:
                pass
            try:
                stratification(2, 3, 1.2)
            except ValueError:
                pass

    Validation
    ----------

        Consider the stratification culture with weight :math:`w` and let
        :math:`m_{\\text{cut}} = \\lfloor w * m \\rfloor`. Then, the probability of generating a
        ranking :math:`\\succ` is 0 if the top :math:`m_{\\text{cut}}` are not
        :math:`0, 1, \\ldots, m_{\\text{cut}}` and otherwise :math:`\\frac{1}{m_{\\text{cut}}!}`.

        .. image:: ../validation_plots/ordinal/stratification_0_2.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a stratification culture with w=0.2

        .. image:: ../validation_plots/ordinal/stratification_0_4.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a stratification culture with w=0.4

        .. image:: ../validation_plots/ordinal/stratification_0_6.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a stratification culture with w=0.6

        Whenever :math:`w=0` or :math:`w=1`, all rankings should be equally likely to be generated.

        .. image:: ../validation_plots/ordinal/stratification_uniform_0.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a stratification culture with w=0.6

        .. image:: ../validation_plots/ordinal/stratification_uniform_1.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a stratification culture with w=0.6

    References
    ----------

        `Putting a compass on the map of elections
        <https://www.ijcai.org/proceedings/2021/9>`_,
        *Boehmer, Niclas, Robert Bredereck, Piotr Faliszewski, Rolf Niedermeier, and Stanisław Szufa*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2021.
    """
    if weight < 0 or 1 < weight:
        raise ValueError(
            f"Incorrect value of weight: {weight}. Value should be in [0, 1]"
        )
    rng = np.random.default_rng(seed)
    votes = []
    upper_class_size = int(weight * num_candidates)
    upper_class_candidates = range(upper_class_size, num_candidates)
    for i in range(num_voters):
        vote = list(rng.permutation(upper_class_size))
        vote += list(rng.permutation(upper_class_candidates))
        votes.append(vote)
    return votes


def stratification_theoretical_distribution(
    num_candidates: int, weight: float, rankings: Iterable[tuple[int]] = None
) -> dict:
    validate_int(num_candidates, lower_bound=0)
    if rankings is None:
        rankings = all_rankings(num_candidates)
    upper_class_size = int(weight * num_candidates)
    upper_class_candidates = set(range(upper_class_size))
    distribution = {}
    for rankings in rankings:
        if set(rankings[:upper_class_size]) == upper_class_candidates:
            distribution[rankings] = 1
        else:
            distribution[rankings] = 0
    normaliser = sum(distribution.values())
    for r in distribution:
        distribution[r] /= normaliser
    return distribution
