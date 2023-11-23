
from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def identity(
    num_voters: int = None,
    num_candidates: int = None,
    p: float = 0.5
) -> list[set]:
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

    Returns
    -------
        list[set]
            Approval votes.

    Raises
    ------
        ValueError
            When `p` not in [0,1] interval.
    """

    if p < 0 or 1 < p:
        raise ValueError(f'Incorrect value of p: {p}. Value should be in [0,1]')

    k = int(p * num_candidates)
    vote = {i for i in range(k)}
    return [vote for _ in range(num_voters)]
