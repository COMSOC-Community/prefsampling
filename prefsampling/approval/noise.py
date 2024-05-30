"""
Noise models are sampling procedures parameterised by a central vote and in which the probability
of generating a given vote is dependent on its distance to the central vote.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import numpy as np

from prefsampling.approval.utils import validate_or_generate_central_vote
from prefsampling.inputvalidators import validate_num_voters_candidates
from prefsampling.combinatorics import comb, powerset


class SetDistance(Enum):
    """
    Constants representing the different types of noise that can be applied to the noise sampler.
    """

    HAMMING = "Hamming distance"
    """
    Hamming distance between s1 and s2, i.e., the number of elements that are either only in s1 or
    only in s2.
    """

    JACCARD = "Jaccard distance"
    """
    Jaccard distance between s1 and s2, i.e., the Hamming distance divided by the size of the union
    of s1 and s2.
    """

    ZELINKA = "Zelinka distance"
    """
    Zelinka distance between s1 and s2, i.e., size of the largest set between s1 and s2 minus the
    size of the intersection.
    """

    BUNKE_SHEARER = "Bunke-Shearer distance"
    """
    Bunke-Shearer distance between s1 and s2, i.e., the Zelinka distance divided by size of the
    largest set between s1 and s2.
    """


class DistanceInfiniteError(ValueError):
    """
    Exception thrown when the distance between two points is infinite., typically when dividing by 0
    for the Bunke-Shearer or the Jaccard distances.
    """

    pass


def _compute_distance(
    distance: SetDistance, size_1: int, size_2: int, size_intersection: int
):
    if distance == SetDistance.HAMMING:
        return size_1 + size_2 - 2 * size_intersection
    if distance == SetDistance.JACCARD:
        size_union = size_1 + size_2 - size_intersection
        if size_union == 0:
            raise DistanceInfiniteError
        return 1 - size_intersection / size_union
    if distance == SetDistance.ZELINKA:
        return max(size_1, size_2) - size_intersection
    if distance == SetDistance.BUNKE_SHEARER:
        largest_size = max(size_1, size_2)
        if largest_size == 0:
            raise DistanceInfiniteError
        else:
            return 1 - size_intersection / largest_size
    raise ValueError(
        "The `distance` argument needs to be one of the constant defined in the "
        "approval.SetDistance enumeration. Choices are: "
        + ", ".join(str(s) for s in SetDistance)
    )


@validate_num_voters_candidates
def noise(
    num_voters: int,
    num_candidates: int,
    phi: float,
    rel_size_central_vote: float,
    distance: SetDistance = SetDistance.HAMMING,
    central_vote: set = None,
    impartial_central_vote: bool = False,
    seed: int = None,
) -> list[set]:
    """
    Generates approval votes under the noise model. This model is parameterised by a central
    vote. Approval ballots are then generated based on their distance to the central vote.
    Specifically, a vote is generated with probability :code:`phi` to the power distance between
    the vote and the central vote.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

    For an analogous sampler generating ordinal ballots, see
    :py:func:`~prefsampling.ordinal.mallows.mallows`.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        phi : float
            Noise model parameter, denoting the noise.
        rel_size_central_vote: float
            The relative size of the central vote, if no central vote is provided, the central
            vote is selected uniformly at random among all subsets of candidates of size
            `⌊rel_size_central_vote * num_candidates⌋`.
        distance : :py:class:`~prefsampling.approval.noise.SetDistance`, default: :py:const:`~prefsampling.approval.noise.SetDistance.HAMMING`
           Distance used to compare a given vote to the central vote.
        central_vote : set
            The central vote. Ignored if :code:`impartial_central_vote = True`.
        impartial_central_vote: bool, default: :code:`False`
            If true, the central vote is sampled from :py:func:`~prefsampling.approval.impartial`
            with the same value for the parameter :code:`p` as passed to this sampler.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.

    Returns
    -------
        list[set]
            Approval votes.

    Examples
    --------

        .. testcode::

            from prefsampling.approval import noise, SetDistance

            # Sample a profile from the noise model with 2 voters and 3 candidates and parameters
            # phi = 0.5, p = 0.2, default distance is SetDistance.HAMMING
            noise(2, 3, 0.5, 0.2)

            # You can give a specific distance
            noise(2, 3, 0.5, 0.2, distance=SetDistance.JACCARD)

            # For reproducibility, you can set the seed.
            noise(2, 3, 0.5, 0.2, seed=157)

            # Parameter phi needs to be in [0, 1]
            try:
                noise(2, 3, 1.2, 0.2)
            except ValueError:
                pass
            try:
                noise(2, 3, -0.2, 0.2)
            except ValueError:
                pass

            # Parameter p needs to be in [0, 1]
            try:
                noise(2, 3, 0.5, 1.2)
            except ValueError:
                pass
            try:
                noise(2, 3, 0.5, -0.2)
            except ValueError:
                pass

    Validation
    ----------

        .. image:: ../validation_plots/approval/noise_0_25.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a noise model with phi=0.25

        .. image:: ../validation_plots/approval/noise_0_5.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a noise model with phi=0.5

        .. image:: ../validation_plots/approval/noise_0_75.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a noise model with phi=0.75

        When :code:`phi` is equal to 0, then a single approval ballot should receive all the
        probability mass.

        .. image:: ../validation_plots/approval/noise_0_0.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a noise model with phi=0

        When :code:`phi` is equal to 1, then we are supposed to obtain a uniform distribution over
        all approval ballots.

        .. image:: ../validation_plots/approval/noise_1_0.png
            :width: 800
            :alt: Observed versus theoretical frequencies for a noise model with phi=1


    References
    ----------

        `Evaluating Approval-Based Multiwinner Voting in Terms of Robustness to Noise
        <https://www.ijcai.org/proceedings/2020/11>`_,
        *Ioannis Caragiannis, Christos Kaklamanis, Nikos Karanikolas and George A. Krimpas*,
        Proceedings of the International Joint Conference on Artificial Intelligence, 2020.

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

    if isinstance(distance, Enum):
        distance = SetDistance(distance.value)
    else:
        distance = SetDistance(distance)

    rng = np.random.default_rng(seed)

    central_vote = tuple(
        validate_or_generate_central_vote(
            num_candidates,
            rel_size_central_vote,
            central_vote,
            impartial_central_vote,
            seed,
        )
    )
    size_central_vote = len(central_vote)
    central_non_vote = tuple(j for j in range(num_candidates) if j not in central_vote)
    size_central_non_vote = len(central_non_vote)

    choices = []
    probabilities = []
    # Prepare buckets
    for num_central in range(size_central_vote + 1):
        num_options_in = comb(size_central_vote, num_central)
        for num_non_central in range(size_central_non_vote + 1):
            num_options_out = comb(size_central_non_vote, num_non_central)

            try:
                exponent = _compute_distance(
                    distance,
                    size_central_vote,
                    num_central + num_non_central,
                    num_central,
                )
                factor = phi**exponent
            except DistanceInfiniteError:
                factor = int(phi == 0)

            num_options = num_options_in * num_options_out * factor

            choices.append((num_central, num_non_central))
            probabilities.append(num_options)
    denominator = sum(probabilities)
    probabilities = [p / denominator for p in probabilities]

    # Sample Votes
    votes = []
    for _ in range(num_voters):
        num_central, num_non_central = rng.choice(choices, p=probabilities)
        vote = set(rng.choice(central_vote, num_central, replace=False))
        vote.update(set(rng.choice(central_non_vote, num_non_central, replace=False)))
        votes.append(vote)

    return votes


def theoretical_distribution(
    num_candidates: int,
    phi: float,
    distance: SetDistance,
    rel_size_central_vote: float,
    central_vote: set = None,
    subsets: Iterable[set[int]] = None,
) -> dict:
    if subsets is None:
        subsets = powerset(range(num_candidates))
    central_vote = validate_or_generate_central_vote(
        num_candidates, rel_size_central_vote, central_vote, False
    )
    res = {
        o: phi
        ** _compute_distance(
            distance, len(central_vote), len(o), len(central_vote.intersection(o))
        )
        for o in subsets
    }
    denominator = sum(res.values())
    for o in res:
        res[o] /= denominator
    return res
