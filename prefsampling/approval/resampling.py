"""
Resampling models generate votes based on resampling procedures on a given central vote.
"""

from __future__ import annotations

from collections.abc import Iterable, Collection

import numpy as np

from prefsampling.approval.utils import validate_or_generate_central_vote
from prefsampling.combinatorics import powerset
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
    Generates approval votes from the resampling model. This model is parameterised by a central
    vote and two parameters :code:`phi` and :code:`rel_size_central_vote`. When generating an
    approval vote, all candidates are considered one after the other. For a given candidate,
    with probability :code:`1 - phi` it is (dis)approved as is the case in the central vote;
    with probability :code:`phi`, it is resampled: approved with probability
    :code:`rel_size_central_vote` and not approved with probability
    :code:`1 - rel_size_central_vote`.

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
            with the same value for the parameter :code:`rel_size_central_vote` as passed to this
            sampler.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    Examples
    --------

        .. testcode::

            from prefsampling.approval import resampling

            # Sample a profile from the resampling model with 2 voters and 3 candidates and
            # parameters phi = 0.5, rel_size_central_vote = 0.2
            resampling(2, 3, 0.5, 0.2)

            # You can also provide the central vote
            resampling(2, 3, 0.5, 0.2, central_vote={0, 1})

            # Or it can be sampled uniformly at random
            resampling(2, 3, 0.5, 0.2, impartial_central_vote=True)

            # For reproducibility, you can set the seed.
            resampling(2, 3, 0.5, 0.2, seed=1657)

            # Parameter phi needs to be in [0, 1]
            try:
                resampling(2, 3, 1.2, 0.2)
            except ValueError:
                pass
            try:
                resampling(2, 3, -0.2, 0.2)
            except ValueError:
                pass

            # Parameter rel_size_central_vote needs to be in [0, 1]
            try:
                resampling(2, 3, 0.5, 1.2)
            except ValueError:
                pass
            try:
                resampling(2, 3, 0.5, -0.2)
            except ValueError:
                pass

    Validation
    ----------

        For the resampling model, there is a known expression for the probability of generating a
        given approval ballot. Indeed, consider a case with :math:`m` candidates, resampling
        parameters :math:`\\phi`, and the central vote :math:`b = \\{0, 1, \\ldots, \\lfloor
        p \\times m \\rfloor\\}`. Then, when generating
        a single ballot, the probability for a candidate :math:`c` to be approved of is:

        .. math::

            \\begin{cases}
            (1 - \\phi) + \\phi \\times p & \\text{if } c \\in b \\\\
            \\phi \\times p & \\text{otherwise}
            \\end{cases}

        From this, since votes are sampled independently, everything is known. We can thus validate
        this model.

        .. image:: ../validation_plots/approval/resampling_0_25.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a resampling model with phi=0.25

        .. image:: ../validation_plots/approval/resampling_0_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a resampling model with phi=0.5

        .. image:: ../validation_plots/approval/resampling_0_75.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a resampling model with phi=0.75

        .. image:: ../validation_plots/approval/resampling_1_0.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a resampling model with phi=1

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
    Generates approval votes from disjoint resampling model. In this model, we first generate
    :code:`num_central_votes` disjoint central votes (they can also be provided). Then, when
    generating a ballot, we select at random one central vote uniformly at random and then use the
    resampling procedure with this central vote (see
    :py:func:`~prefsampling.approval.resampling.resampling` for the details). This is essentially a
    mixture of several resampling samplers with different disjoint central votes.

    The procedure to generate the central votes is as follows. Each central vote will be of size
    `⌊rel_size_central_vote * num_candidates⌋`. We uniformly at random partition the candidates into
    :code:`num_central_votes` parts of such size. Note that this implies that some candidates can
    appear in no groups. Moreover, if `rel_size_central_vote * num_central_votes > 1`, then the
    model is not well-defined. If :code:`impartial_central_votes == False` and
    :code:`central_votes is None`, then the central votes are always:
    `{0, 1, ..., s - 1}, {s, s + 1, ..., 2s - 1}, ...` where `s` is equal to
    `⌊rel_size_central_vote * num_candidates⌋`.

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
            The relative size of the central vote. If a single value is given, it is used for all
            groups, otherwise, one value per group needs to be provided.
        num_central_votes : int, default: :code:`None`
            The number of central votes.
        central_votes : Collection[Collection[int]], default: :code:`None`
            The central votes of the different groups. If this parameter is not used, the central
            votes are generated randomly.
        impartial_central_votes : bool, default: :code:`False`
            If set to :code:`True`, then the central votes are generated at random following a
            uniform distribution over all suitable collection of central votes.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    Examples
    --------

        .. testcode::

            from prefsampling.approval import disjoint_resampling

            # Sample a profile from the disjoint resampling model with 2 voters and
            # 3 candidates and parameters phi = 0.5, rel_size_central_vote = 0.2
            # and 2 central votes (here {0} and {1})
            disjoint_resampling(2, 3, 0.5, 0.2, num_central_votes=2)

            # The central votes can be sampled uniformly at random
            disjoint_resampling(2, 3, 0.5, 0.2, num_central_votes=2, impartial_central_votes=True)

            # Or they can be provided.
            disjoint_resampling(2, 3, 0.5, 0.2, central_votes=({0, 1}, {2}))

            # Don't forget that they have to be disjoint
            try:
                disjoint_resampling(2, 3, 0.5, 0.2, central_votes=({0, 1}, {1, 2}))
            except ValueError:
                pass

            # You need to use either num_central_votes (with or without impartial_central_votes)
            # or central_votes
            try:
                disjoint_resampling(2, 3, 0.5, 0.2)
            except ValueError:
                pass

            # For reproducibility, you can set the seed.
            disjoint_resampling(2, 3, 0.5, 0.2, num_central_votes=2, seed=1657)

            # Parameter phi needs to be in [0, 1]
            try:
                disjoint_resampling(2, 3, 1.5, 0.2, num_central_votes=2)
            except ValueError:
                pass
            try:
                disjoint_resampling(2, 3, -0.5, 0.2, num_central_votes=2)
            except ValueError:
                pass

            # Parameter rel_size_central_vote needs to be in [0, 1]
            try:
                disjoint_resampling(2, 3, 0.5, 1.2, num_central_votes=2)
            except ValueError:
                pass
            try:
                disjoint_resampling(2, 3, 0.5, -0.2, num_central_votes=2)
            except ValueError:
                pass

    Validation
    ----------

        For the disjoint resampling model, since it basically consists of a mixture of
        resampling models, the probability distribution over the outcome is known: is it a linear
        combination of the probability of the mixed resampling models (see
        :py:func:`~prefsampling.approval.resampling.resampling` for the details of the latter).
        We can thus validate this model.

        .. image:: ../validation_plots/approval/disjoint_resampling_0_25.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a disjoint resampling model phi=0.25

        .. image:: ../validation_plots/approval/disjoint_resampling_0_33.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a disjoint resampling model phi=0.33

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
            f"Incorrect value of rel_size_central_vote: {rel_size_central_vote}. Value should be "
            f"in [0,1]"
        )

    rng = np.random.default_rng(seed)

    if central_votes is None:
        if num_central_votes is None:
            raise ValueError(
                "If you do not provide the central votes for the disjoint resampling model, you "
                "need to provide the number of groups."
            )

        validate_int(
            num_central_votes, value_descr="number of central votes", lower_bound=1
        )

        if rel_size_central_vote * num_central_votes > 1:
            raise ValueError(
                f"For the disjoint resampling model we need rel_size_central_vote * "
                f"num_central_votes <= 1 as otherwise there would not be enough candidates."
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
            for cand in central_vote:
                if cand in seen_candidates:
                    raise ValueError(
                        "In the disjoint resampling model the central votes need to be disjoint. "
                        f"Currently, candidate {cand} appears in at least 2 central votes."
                    )
                seen_candidates.add(cand)
        if len(seen_candidates) > num_candidates:
            raise ValueError(
                "There are more candidates appearing in the central votes than the number of "
                "candidates provided as an argument."
            )
        if max(seen_candidates) >= num_candidates:
            raise ValueError(
                "The candidates need to be called 0, 1, ..., num_candidates, this is not the case "
                "in the provided central votes."
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
    rel_size_central_vote: float,
    num_legs: int,
    central_vote: set = None,
    impartial_central_vote: bool = False,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes from moving resampling model. In the moving resampling model, the
    ballot of the first voter is always the central vote, other voters are grouped in so-called
    legs. Within a leg, votes are generated one after another using the resampling procedure
    (see :py:func:`~prefsampling.approval.resampling.resampling` for the details) using as central
    vote the ballot of the previous voter (the first voter for the first voter of a leg).

    Note that for a given number of voters, votes are not sampled independently.

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
        num_legs : int
            The number of legs.
        central_vote : set
            The central vote. Ignored if :code:`impartial_central_vote = True`.
        impartial_central_vote: bool, default: :code:`False`
            If true, the central vote is sampled from :py:func:`~prefsampling.approval.impartial`
            with the same value for the parameter :code:`rel_size_central_vote` as passed to this
            sampler.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    Examples
    --------

        .. testcode::

            from prefsampling.approval import moving_resampling

            # Sample a profile from the moving resampling model with 4 voters and 3 candidates and
            # parameters phi = 0.5, rel_size_central_vote = 0.2 and 2 legs
            moving_resampling(4, 3, 0.5, 0.2, 2)

            # You can also provide the central vote
            moving_resampling(4, 3, 0.5, 0.2, 2, central_vote={0, 1})

            # Or it can be sampled uniformly at random
            moving_resampling(4, 3, 0.5, 0.2, 2, impartial_central_vote=True)

            # For reproducibility, you can set the seed.
            moving_resampling(4, 3, 0.5, 0.2, 2, seed=1657)

            # Parameter phi needs to be in [0, 1]
            try:
                moving_resampling(4, 3, 1.5, 0.2, 2)
            except ValueError:
                pass
            try:
                moving_resampling(4, 3, -0.5, 0.2, 2)
            except ValueError:
                pass

            # Parameter rel_size_central_vote needs to be in [0, 1]
            try:
                moving_resampling(4, 3, 0.5, 1.2, 2)
            except ValueError:
                pass
            try:
                moving_resampling(4, 3, 0.5, -0.2, 2)
            except ValueError:
                pass

    Validation
    ----------

        There is no known expression for the probability distribution governing moving resampling
        models.

        .. image:: ../validation_plots/approval/moving_resampling_0_25.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a moving resampling model phi=0.25

        .. image:: ../validation_plots/approval/moving_resampling_0_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a moving resampling model phi=0.5

        .. image:: ../validation_plots/approval/moving_resampling_0_75.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a moving resampling model phi=0.75

        .. image:: ../validation_plots/approval/moving_resampling_1_0.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a moving resampling model phi=1


    References
    ----------
        None.
    """
    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0,1]")

    if rel_size_central_vote < 0 or 1 < rel_size_central_vote:
        raise ValueError(
            f"Incorrect value of p: {rel_size_central_vote}. Value should be in [0,1]"
        )

    validate_int(num_legs, "number of legs", lower_bound=1)

    if num_legs > num_voters:
        raise ValueError("The number of legs cannot exceed the number of voters.")

    rng = np.random.default_rng(seed)

    central_vote = validate_or_generate_central_vote(
        num_candidates,
        rel_size_central_vote,
        central_vote,
        impartial_central_vote,
        seed,
    )

    breaking_points = [int(num_voters / num_legs) * i for i in range(num_legs)]

    votes = [central_vote]
    for v in range(1, num_voters):
        vote = set()
        for c in range(num_candidates):
            if rng.random() <= phi:
                if rng.random() <= rel_size_central_vote:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes.append(vote)
        central_vote = vote
        if v in breaking_points:
            central_vote = votes[0]
    return votes
