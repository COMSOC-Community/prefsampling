"""
Resampling models generate votes based on resampling procedures on a given central vote.
"""

from __future__ import annotations

import copy

import math
from collections.abc import Iterable, Collection

import numpy as np

from prefsampling.approval.utils import validate_or_generate_central_vote
from prefsampling.combinatorics import random_partition, powerset
from prefsampling.inputvalidators import validate_num_voters_candidates, validate_int


def _resample_vote(
    num_candidates: int,
    phi: float,
    rel_size_central_vote: float,
    central_vote: set[int],
    rng,
) -> set[int]:
    """
    Generates a single vote following the resampling procedure.
    """
    vote = set()
    for c in range(num_candidates):
        if rng.random() <= phi:
            if rng.random() <= rel_size_central_vote:
                vote.add(c)
        else:
            if c in central_vote:
                vote.add(c)
    return vote


@validate_num_voters_candidates
def resampling(
    num_voters: int,
    num_candidates: int,
    phi: float,
    rel_size_central_vote: float,
    central_vote: set = None,
    impartial_central_vote: bool = False,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes from the resampling model. This model is parameterised by a central vote and two parameters
    :code:`phi` and :code:`rel_size_central_vote`. When generating an approval vote, all candidates are considered one
    after the other. For a given candidate, with probability :code:`1 - phi` it is (dis)approved as is the case in the
    central vote; with probability :code:`phi`, it is resampled: approved with probability :code:`rel_size_central_vote`
    and not approved with probability :code:`1 - rel_size_central_vote`.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        phi : float
            The resampling probability, guiding the dispersion.
        rel_size_central_vote : float
            The relative size of the central vote.
        central_vote : set
            The central vote. Ignored if :code:`impartial_central_vote = True`.
        impartial_central_vote: bool, default: :code:`False`
            If true, the central vote is sampled from :py:func:`~prefsampling.approval.impartial`
            with the same value for the parameter :code:`rel_size_central_vote` as passed to this sampler.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    References
    ----------

        `How to Sample Approval Elections?
        <https://www.ijcai.org/proceedings/2022/71>`_,
        *Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
        Krzysztof Sornat and Nimrod Talmon*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2022.
    """

    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0,1]")

    if rel_size_central_vote < 0 or 1 < rel_size_central_vote:
        raise ValueError(
            f"Incorrect value of p: {rel_size_central_vote}. Value should be in [0,1]"
        )

    rng = np.random.default_rng(seed)

    central_vote = validate_or_generate_central_vote(
        num_candidates,
        rel_size_central_vote,
        central_vote,
        impartial_central_vote,
        seed,
    )

    votes = []
    for v in range(num_voters):
        votes.append(
            _resample_vote(
                num_candidates, phi, rel_size_central_vote, central_vote, rng
            )
        )
    return votes


def resampling_theoretical_distribution(
    num_candidates: int,
    phi: float,
    rel_size_central_vote: float,
    central_vote: set = None,
    subsets: Iterable[set[int]] = None,
) -> dict:
    if subsets is None:
        subsets = powerset(range(num_candidates))
    central_vote = validate_or_generate_central_vote(
        num_candidates, rel_size_central_vote, central_vote, False
    )
    distribution = {}
    for outcome in subsets:
        prob = 1
        for c in range(num_candidates):
            if c in central_vote and c in outcome:
                prob *= (1 - phi) + phi * rel_size_central_vote
            elif c in central_vote and c not in outcome:
                prob *= phi * (1 - rel_size_central_vote)
            elif c not in central_vote and c in outcome:
                prob *= phi * rel_size_central_vote
            else:
                prob *= (1 - phi) + phi * (1 - rel_size_central_vote)
        distribution[outcome] = prob
    return distribution


@validate_num_voters_candidates
def disjoint_resampling(
    num_voters: int,
    num_candidates: int,
    phi: float,
    rel_size_central_vote: float | Iterable[float],
    num_central_votes: int = None,
    central_votes: Collection[Collection[int]] = None,
    impartial_central_votes: bool = False,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes from disjoint resampling model. In this model, we first generate :code:`num_groups`
    disjoint central votes (they can also be provided). Then, when generating a ballot, we select at random one central
    vote uniformly at random and then use the resampling procedure with this central vote (see
    :py:func:`~prefsampling.approval.resampling.resampling` for the details). This is essentially a mixture of several
    resampling samplers with different disjoint central votes.

    The procedure to generate the central votes is as follows. Each central vote will be of size
    `⌊rel_size_central_vote * num_candidates⌋`. We uniformly at random partition the candidates into :code:`num_groups`
    parts of such size. Note that this implies that some candidates can appear in no groups. Moreover, if
    `rel_size_central_vote * num_central_votes > 1`, then the model is not well-defined.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        phi : float
            Disjoint resampling model parameter, denoting the noise.
        rel_size_central_vote : float | Iterable[float]
            The relative size of the central vote. If a single value is given, it is used for all groups, otherwise, one
            value per group needs to be provided.
        num_central_votes : int, default: 2
            The number of central votes.
        central_votes : Collection[Collection[int]]
            The central votes of the different groups. If this parameter is not used, the central votes are generated
            randomly.
        impartial_central_votes : bool, default: :code:`False`
            If set to :code:`True`, then the central votes are generated at random following a uniform distribution
            over all suitable collection of central votes.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    References
    ----------

        `How to Sample Approval Elections?
        <https://www.ijcai.org/proceedings/2022/71>`_,
        *Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
        Krzysztof Sornat and Nimrod Talmon*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2022.
    """

    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0,1]")

    if rel_size_central_vote < 0 or 1 < rel_size_central_vote:
        raise ValueError(
            f"Incorrect value of rel_size_central_vote: {rel_size_central_vote}. Value should be in [0,1]"
        )

    rng = np.random.default_rng(seed)

    if central_votes is None:
        if num_central_votes is None:
            raise ValueError(
                "If you do not provide the central votes for the disjoint resampling model, you need to "
                "provide the number of groups."
            )

        validate_int(
            num_central_votes, value_descr="number of central votes", lower_bound=1
        )

        if rel_size_central_vote * num_central_votes > 1:
            raise ValueError(
                f"For the disjoint resampling model we need rel_size_central_vote * num_central_votes <= 1 as "
                f"otherwise there would not be enough candidates."
            )

        central_votes_size = int(rel_size_central_vote * num_candidates)
        if impartial_central_votes:
            central_votes = rng.choice(
                range(num_candidates), size=(num_central_votes, central_votes_size)
            )
        else:
            central_votes = [
                {g * central_votes_size + i for i in range(central_votes_size)}
                for g in range(num_central_votes)
            ]
    else:
        seen_candidates = set()
        for central_vote in central_votes:
            for alt in central_vote:
                if alt in seen_candidates:
                    raise ValueError(
                        "In the disjoint resampling model, the central votes need to be disjoint. "
                        f"Currently, candidate {alt} appears in at least 2 central votes."
                    )
                seen_candidates.add(alt)
        if len(seen_candidates) > num_candidates:
            raise ValueError(
                "There are more candidates appearing in the central votes than the number of candidates "
                "provided as an argument."
            )
        if max(seen_candidates) >= num_candidates:
            raise ValueError(
                "The candidates need to be called 0, 1, ..., num_candidates, this is not the case in the "
                "provided central votes."
            )

    votes = []
    for v in range(num_voters):
        central_vote = rng.choice(central_votes)
        votes.append(
            _resample_vote(
                num_candidates, phi, rel_size_central_vote, central_vote, rng
            )
        )
    return votes


def disjoint_resampling_theoretical_distribution(
    num_candidates: int,
    phi: float,
    rel_size_central_vote: float,
    num_central_votes: int,
    central_votes: Iterable[set] = None,
    subsets: Iterable[set[int]] = None,
) -> dict:
    if subsets is None:
        subsets = powerset(range(num_candidates))
    if central_votes is None:
        central_votes_size = int(rel_size_central_vote * num_candidates)
        central_votes = [
            {g * central_votes_size + i for i in range(central_votes_size)}
            for g in range(num_central_votes)
        ]
    distribution = {s: 0 for s in subsets}
    for central_vote in central_votes:
        local_distr = resampling_theoretical_distribution(
            num_candidates,
            phi,
            rel_size_central_vote,
            central_vote=central_vote,
            subsets=subsets,
        )
        for s, d in local_distr.items():
            distribution[s] += d / len(central_votes)
    return distribution


@validate_num_voters_candidates
def moving_resampling(
    num_voters: int,
    num_candidates: int,
    phi: float,
    p: float,
    num_legs: int = 1,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes from moving resampling model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        phi : float
            Moving resampling model parameter, denoting the noise.
        p : float
            Moving resampling model parameter, denoting the length of central vote.
        num_legs : int, default: 1
            Moving resampling model parameter, denoting the number of legs.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    References
    ----------

        `How to Sample Approval Elections?
        <https://www.ijcai.org/proceedings/2022/71>`_,
        *Stanisław Szufa, Piotr Faliszewski, Łukasz Janeczko, Martin Lackner, Arkadii Slinko,
        Krzysztof Sornat and Nimrod Talmon*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2022.
    """
    validate_int(num_legs, "number of legs", lower_bound=1)

    rng = np.random.default_rng(seed)

    breaks = [int(num_voters / num_legs) * i for i in range(num_legs)]

    k = math.floor(p * num_candidates)
    central_vote = {i for i in range(k)}
    ccc = copy.deepcopy(central_vote)

    votes = [set() for _ in range(num_voters)]
    votes[0] = copy.deepcopy(central_vote)

    for v in range(1, num_voters):
        vote = set()
        for c in range(num_candidates):
            if rng.random() <= phi:
                if rng.random() <= p:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote
        central_vote = copy.deepcopy(vote)

        if v in breaks:
            central_vote = copy.deepcopy(ccc)

    return votes
