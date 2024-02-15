from collections.abc import Callable

import numpy as np


def mixture(
    num_voters: int,
    num_candidates: int,
    samplers: list[Callable],
    weights: list[float],
    sampler_parameters: list[dict],
    seed: int = None,
):
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
        samplers: list
            List of samplers.
        weights: list
            Probability distribution over the samplers, the sampler in position k has weight :code:`weights[k]`.
        sampler_parameters: list[dict]
            List of dictionaries passed as keyword parameters of the samplers. Number of voters and
            number of candidates of these dictionaries are not taken into account.
        seed : int
            Seed for numpy random number generator.
            Note that this is only the seed for this function.
            If you want to use particular seed for the functions generating votes,
            you should pass it as parameter within the :code:`sampler_parameters` list.
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
) -> np.ndarray:
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
        np.ndarray
            Ordinal votes.
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

    all_votes = None
    for num_voters, sampler, params in zip(
        num_voters_per_sampler, samplers, sampler_parameters
    ):
        if num_voters > 0:
            votes = sampler(**params)
            if all_votes is None:
                all_votes = votes
            elif isinstance(all_votes, np.ndarray):
                all_votes = np.concatenate((all_votes, votes), axis=0)
            elif isinstance(all_votes, list):
                all_votes.extend(votes)
            else:
                raise ValueError(
                    f"The type of the votes returned by the sampler "
                    f"{sampler.__name__} do not match known types and thus cannot "
                    f"be used for concatenation of samplers. Did you mix-up ordinal "
                    f"and approval samplers?"
                )
    return all_votes
