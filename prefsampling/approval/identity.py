from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def identity(
    num_voters: int, num_candidates: int, p: float = 0.5, seed: int = None
) -> list[set[int]]:
    """
    Generates approval votes from identity culture.

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    p : float, default: 0.5
        Resampling model parameter, denoting the average vote length.
    seed : int
        Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    Raises
    ------
        ValueError
            When `p` not in [0,1] interval.
    """

    if p < 0 or 1 < p:
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0,1]")

    k = int(p * num_candidates)
    vote = {i for i in range(k)}
    return [vote for _ in range(num_voters)]


@validate_num_voters_candidates
def full(num_voters: int, num_candidates: int, seed: int = None) -> list[set[int]]:
    """
    Generates approval votes where each voter approves all the candidates.

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    seed : int
        Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """

    vote = {i for i in range(num_candidates)}
    return [vote for _ in range(num_voters)]


@validate_num_voters_candidates
def empty(num_voters: int, num_candidates: int, seed: int = None) -> list[set[int]]:
    """
    Generates approval votes where each vote is empty.

    Parameters
    ----------
    num_voters : int
        Number of Voters.
    num_candidates : int
        Number of Candidates.
    seed : int
        Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.
    """
    return [set() for _ in range(num_voters)]
