import numpy as np


def mixture(
        num_voters: int,
        num_candidates: int,
        weights: list,
        list_of_cultures: list,
        list_of_params: list,
        seed: int = None,
) -> list[set[int]]:
    """
    Generates a mixture of approval votes from different cultures.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        weights: list
            Probability of being sampled from a given culture
        list_of_cultures: list,
            List of the cultures.
        list_of_params: list,
            List of params for each considered culture
            (excluding the number of voters and the number of candidates).
        seed : int
            Seed for numpy random number generator.
            Note that this is only the seed for this function.
            If you want to use particular seed for the functions generating votes,
            you should pass it as argument within the list of params.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """
    rng = np.random.default_rng(seed)

    options = [i for i in range(len(weights))]
    sum_weights = sum(weights)
    norm_weights = [w / sum_weights for w in weights]
    samples = list(rng.choice(options, size=num_voters, replace=True, p=norm_weights))
    list_of_num_voters = [samples.count(i) for i in range(len(weights))]
    return concat(list_of_num_voters, num_candidates, list_of_cultures, list_of_params)


def concat(
        list_of_num_voters: list,
        num_candidates: int,
        list_of_cultures: list,
        list_of_params: list
) -> list[set[int]]:
    """
    Generates a concatenation of approval votes from different cultures.

    Parameters
    ----------
        list_of_num_voters : int
            List of numbers of voters to be sampled from each culture.
        num_candidates : int
            Number of Candidates.
        list_of_cultures: list,
            List of the cultures.
        list_of_params: list,
            List of params for each considered culture
            (excluding the number of voters and the number of candidates).

    Returns
    -------
        list[set[int]]
            Approval votes.
    """
    all_votes = []
    for num_voters, culture, params in zip(list_of_num_voters, list_of_cultures, list_of_params):
        if num_voters > 0:
            votes = culture(num_voters, num_candidates, **params)
            for vote in votes:
                all_votes.append(vote)
    return all_votes
