from collections.abc import Callable

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def truncated_ordinal(
    num_voters: int,
    num_candidates: int,
    p: float,
    ordinal_sampler: Callable,
    ordinal_sampler_parameters: dict,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes by truncating ordinal votes sampled from a given ordinal sampler.

    The process is as follows: ordinal votes are sampled from the ordinal sampler. These votes are
    then truncated in a way that each approval vote consists in the `p * num_candidates` first
    candidates of the ordinal vote.

    Parameters
    ----------
        num_voters: int
            Number of voters
        num_candidates: int
            Number of candidates
        p: float,
            Ratio of approved candidates.
        ordinal_sampler: Callable
            The ordinal sampler to be used.
        ordinal_sampler_parameters: dict
            The arguments passed ot the ordinal sampler. The num_voters, num_candidates and seed
            parameters are overriden by those passed to this function.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes
    """
    if p < 0 or 1 < p:
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0,1]")

    ordinal_sampler_parameters["num_voters"] = num_voters
    ordinal_sampler_parameters["num_candidates"] = num_candidates
    ordinal_sampler_parameters["seed"] = seed
    ordinal_votes = ordinal_sampler(**ordinal_sampler_parameters)

    vote_length = int(p * num_candidates)
    return [{int(c) for c in vote[0:vote_length]} for vote in ordinal_votes]
