import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def single_peaked_conitzer(
    num_voters: int, num_candidates: int, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes that are single-peaked following the distribution defined by Conitzer
    (2009). The preferences generated are single-peaked with respect to the axis `0, 1, 2, ...`.
    Votes are generated uniformly at random as follows. The most preferred candidate (the peak)
    is selected uniformly at random. Then, either the candidate on the left, or the one on the
    right of the peak comes second in the ordering. Each case occurs with probability 0.5.
    The method is then iterated for the next left and right candidates (only one of them being
    different from before).

    This method ensures that the probability for a given candidate to be the peak is uniform
    (as opposed to the method :py:func:`~prefsampling.ordinal.single_peaked_walsh`).

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
    np.ndarray
        Ordinal votes.
    """
    rng = np.random.default_rng(seed)
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    for i in range(num_voters):
        peak = rng.choice(range(num_candidates))
        votes[i][0] = peak
        left = peak - 1
        right = peak + 1
        for j in range(1, num_candidates):
            # If we are stuck on the left or on the right, we fill in the vote
            if left < 0:
                votes[i][j:] = range(right, num_candidates)
                break
            if right >= num_candidates:
                votes[i][j:] = range(left, -1, -1)
                break
            if rng.random() < 0.5:
                votes[i][j] = right
                right += 1
            else:
                votes[i][j] = left
                left -= 1

    return votes


@validate_num_voters_candidates
def single_peaked_circle(
    num_voters: int, num_candidates: int, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes that are single-peaked on a circle following a distribution inspired
    from the one by Conitzer (2009) for single-peakedness on a line (see
    :py:func:`~prefsampling.ordinal.single_peaked_conitzer`). This method starts by
    determining the most preferred candidate (the peak). This is done with uniform probability
    over the candidates. Then, subsequent positions in the ordering are filled by taking either the
    next available candidate on the left or on the right, both cases occuring with probability 0.5.
    Left and right are defined here in terms of the circular axis: 0, 1, ..., m, 1.

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
    np.ndarray
        Ordinal votes.
    """
    rng = np.random.default_rng(seed)
    votes = np.zeros([num_voters, num_candidates], dtype=int)
    for j in range(num_voters):
        votes[j][0] = rng.choice(range(num_candidates))
        left = votes[j][0] - 1
        left %= num_candidates
        right = votes[j][0] + 1
        right %= num_candidates
        for k in range(1, num_candidates):
            if rng.random() < 0.5:
                votes[j][k] = left
                left -= 1
                left %= num_candidates
            else:
                votes[j][k] = right
                right += 1
                right %= num_candidates
    return votes


@validate_num_voters_candidates
def single_peaked_walsh(
    num_voters: int, num_candidates: int, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes that are single-peaked following the process described by Walsh (2015).
    The votes are generated from least preferred to most preferred candidates. A given position
    in the ordering is filled by selecting with uniform probability either the leftmost or the
    rightmost candidates that have not yet been positioned in the vote (defined by the axis
    0, 1, 2, ...).

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
    np.ndarray
        Ordinal votes.
    """
    rng = np.random.default_rng(seed)
    votes = np.zeros([num_voters, num_candidates], dtype=int)

    for j in range(num_voters):
        _single_peaked_walsh_recursor(
            0, num_candidates - 1, votes[j], num_candidates - 1, rng
        )

    return votes.astype(int)


def _single_peaked_walsh_recursor(
    a: int, b: int, vote: np.ndarray, position: int, rng: np.random.Generator
) -> None:
    """
    Function that implements the recursor needed for sampling preferences that are single-peaked
    following the process described by Walsh (2015). Populates the vote by side effect.

    Parameters
    ----------
    a: int
        The leftmost candidate
    b: int
        The rightmost candidate
    vote: np.ndarray
        The ballot that will be filled up by side effect.
    position: int
        The position to place the candidate in the ballot.
    rng: np.random.Generator
        The random number generator used to perform random operations.
    """
    if position == -1:
        return
    elif rng.random() < 0.5:
        vote[position] = a
        _single_peaked_walsh_recursor(a + 1, b, vote, position - 1, rng)
    else:
        vote[position] = b
        _single_peaked_walsh_recursor(a, b - 1, vote, position - 1, rng)
