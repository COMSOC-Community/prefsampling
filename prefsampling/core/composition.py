from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


def mixture(
    num_voters: int,
    num_candidates: int,
    samplers: list[Callable],
    weights: list[float],
    sampler_parameters: list[dict],
    seed: int = None,
) -> list:
    """
    Generates a mixture of samplers. The process works as follows: for each vote, we sample which
    sample will be used to generate it based on the weight distribution of the samplers,
    then, the corresponding number of votes are sampled from the samplers and concatenated together
    to form the final set of votes.

    Note that votes are not sampled one after the other from the samplers but all at once. This
    is important if you are using samplers that are not independent.

    It is assumed that you pass samplers that are all about the same type of ballots (only ordinal
    or only approval for instance). If you don't, the code will probably fail.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        samplers: list[Callable]
            List of samplers.
        weights: list[float]
            Probability distribution over the samplers, the sampler in position k has weight
            :code:`weights[k]`.
        sampler_parameters: list[dict]
            List of dictionaries passed as keyword parameters of the samplers. Number of voters and
            number of candidates of these dictionaries are not taken into account.
        seed : int, default: :code:`None`
            Seed for numpy random number generator.
            Note that this is only the seed for this function.
            If you want to use particular seed for the functions generating votes,
            you should pass it as parameter within the :code:`sampler_parameters` list.

    Returns
    -------
        list
            The votes sampled from the mixture.

    Examples
    --------

        .. testcode::

            from prefsampling.core import mixture
            from prefsampling.ordinal import mallows

            # A mixture of two Mallows' models with different phi and central votes.
            # The first model has weight 0.7 and the second 0.3.
            # There are 10 voters and 5 candidates.

            mixture(
                10,
                5,
                [mallows, mallows],
                [0.7, 0.3],
                [
                    {'phi': 0.2, 'central_vote': range(5)},
                    {'phi': 0.9, 'central_vote': [4, 3, 2, 1, 0]}
                ],
            )

            # The weights are re-normalised if they don't add up to one.
            # The real weights here would be 3/4, 1/4.

            from prefsampling.approval import noise, identity

            mixture(
                10,
                5,
                [noise, identity],
                [0.3, 0.1],
                [
                    {'rel_size_central_vote': 0.2, 'phi': 0.4},
                    {'rel_num_approvals': 0.6}
                ],
            )
    """
    if len(samplers) != len(weights):
        raise ValueError(
            "For a mixture, you need to provide one weight per sampler, no more, no "
            "less."
        )
    if len(samplers) != len(sampler_parameters):
        raise ValueError(
            "For a mixture, you need to provide one dictionary of parameters per "
            "sampler, no more, no less."
        )
    if min(weights) < 0:
        raise ValueError("For a mixture, the weight of a sampler cannot be negative.")
    if sum(weights) == 0:
        raise ValueError("For a mixture, the sum of the weights cannot be 0.")

    rng = np.random.default_rng(seed)

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    num_samplers = len(samplers)
    samples = rng.choice(range(num_samplers), size=num_voters, replace=True, p=weights)
    counts = np.unique(samples, return_counts=True)
    num_voters_per_sampler = [0 for _ in range(num_samplers)]
    for i, j in enumerate(counts[0]):
        num_voters_per_sampler[j] = counts[1][i]
    return concatenation(
        num_voters_per_sampler, num_candidates, samplers, sampler_parameters
    )


def concatenation(
    num_voters_per_sampler: list[int],
    num_candidates: int,
    samplers: list[Callable],
    sampler_parameters: list[dict],
) -> list:
    """
    Generate votes from different samplers and concatenate them together to form the final set of
    votes.

    Note that votes are not sampled one after the other from the samplers but all at once. This
    is important if you are using samplers that are not independent.

    It is assumed that you pass samplers that are all about the same type of ballots (only ordinal
    or only approval for instance). If you don't, the code will probably fail.

    Parameters
    ----------
        num_voters_per_sampler : int
            List of numbers of voters to be sampled from each sampler.
        num_candidates : int
            Number of Candidates.
        samplers: list,
            List of samplers.
        sampler_parameters: list,
            List of dictionaries passed as keyword parameters of the samplers. Number of voters and
            number of candidates of these dictionaries are not taken into account.

    Returns
    -------
        list
            The concatenated votes.

    Examples
    --------

        .. testcode::

            from prefsampling.core import concatenation
            from prefsampling.ordinal import mallows

            # A concatenation of two Mallows' models with different phi and central votes.
            # 4 votes are sampled from the first model and 6 votes from the second.
            # There are 5 candidates

            concatenation(
                [4, 6],
                5,
                [mallows, mallows],
                [
                    {'phi': 0.2, 'central_vote': range(5)},
                    {'phi': 0.9, 'central_vote': [4, 3, 2, 1, 0]}
                ],
            )
    """

    if len(num_voters_per_sampler) != len(samplers):
        raise ValueError(
            "For a concatenation, you need to provide one number of voters per "
            "sampler, no more, no less."
        )
    if len(samplers) != len(sampler_parameters):
        raise ValueError(
            "For a concatenation, you need to provide one dictionary of parameters"
            " per sampler, no more, no less."
        )

    for i, params in enumerate(sampler_parameters):
        params["num_voters"] = num_voters_per_sampler[i]
        params["num_candidates"] = num_candidates

    all_votes = []
    for num_voters, sampler, params in zip(
        num_voters_per_sampler, samplers, sampler_parameters
    ):
        if num_voters > 0:
            new_votes = sampler(**params)
            if not isinstance(new_votes, Iterable):
                raise ValueError(
                    f"The sampler {samplers} did not return an iterable, we cannot "
                    f"concatenate."
                )
            all_votes.extend(new_votes)
    return all_votes
