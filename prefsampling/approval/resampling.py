import copy

import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def resampling(
    num_voters: int,
    num_candidates: int,
    phi: float,
    p: float,
    seed: int = None,
    central_vote: set = None,
) -> list[set[int]]:
    """
    Generates approval votes from the resampling model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        phi : float
            Resampling model parameter, denoting the noise.
        p : float
            Resampling model parameter, denoting the average vote length.
        seed : int
            Seed for numpy random number generator.
        central_vote : set
            The central vote.

    Returns
    -------
        list[set[int]]
            Approval votes.

    Raises
    ------
        ValueError
            When `phi` not in [0,1] interval.
            When `p` not in [0,1] interval.
    """

    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0,1]")

    if p < 0 or 1 < p:
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0,1]")

    rng = np.random.default_rng(seed)

    k = int(p * num_candidates)
    if central_vote is None:
        central_vote = set(range(k))

    votes = [set() for _ in range(num_voters)]
    for v in range(num_voters):
        vote = set()
        for c in range(num_candidates):
            if rng.random() <= phi:
                if rng.random() <= p:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote

    return votes


@validate_num_voters_candidates
def disjoint_resampling(
    num_voters: int,
    num_candidates: int,
    phi: float,
    p: float,
    g: int = 2,
    seed: int = None,
) -> list[set[int]]:
    """
    Generates approval votes from disjoint resampling model.

    Parameters
    ----------
        num_voters : int
            Number of Voters.
        num_candidates : int
            Number of Candidates.
        phi : float
            Disjoint resampling model parameter, denoting the noise.
        p : float
            Disjoint resampling model parameter, denoting the length of central vote.
        g : int, default: 2
            Disjoint resampling model parameter, denoting the number of groups.
        seed : int
            Seed for numpy random number generator.

    Returns
    -------
        list[set[int]]
            Approval votes.

    Raises
    ------
        ValueError
            When `phi` not in [0,1] interval.
            When `p` not in [0,1] interval.
    """

    if phi < 0 or 1 < phi:
        raise ValueError(f"Incorrect value of phi: {phi}. Value should be in [0,1]")

    if p < 0 or 1 < p:
        raise ValueError(f"Incorrect value of p: {p}. Value should be in [0,1]")

    if p * g > 1:
        raise ValueError(f"Disjoint model is not well defined when p * g > 1")

    rng = np.random.default_rng(seed)

    num_groups = g
    k = int(p * num_candidates)

    votes = [set() for _ in range(num_voters)]

    central_votes = []
    for g in range(num_groups):
        central_votes.append({g * k + i for i in range(k)})

    for v in range(num_voters):
        central_vote = rng.choice(central_votes)

        vote = set()
        for c in range(num_candidates):
            if rng.random() <= phi:
                if rng.random() <= p:
                    vote.add(c)
            else:
                if c in central_vote:
                    vote.add(c)
        votes[v] = vote

    return votes


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

    Raises
    ------
        ValueError
            When `phi` not in [0,1] interval.
            When `p` not in [0,1] interval.
    """

    rng = np.random.default_rng(seed)

    breaks = [int(num_voters / num_legs) * i for i in range(num_legs)]

    k = int(p * num_candidates)
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
