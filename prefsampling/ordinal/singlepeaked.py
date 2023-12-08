import numpy as np

from prefsampling.inputvalidators import validate_num_voters_candidates


@validate_num_voters_candidates
def single_peaked_conitzer(
    num_voters: int, num_candidates: int, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes that are single-peaked following the distribution defined by
    `Conitzer (2009) <https://arxiv.org/abs/1401.3449>`_. The preferences generated are
    single-peaked with respect to the axis `0, 1, 2, ...`. Votes are generated uniformly at random
    as follows. The most preferred candidate (the peak) is selected uniformly at random. Then,
    either the candidate on the left, or the one on the right of the peak comes second in the
    ordering. Each case occurs with probability 0.5. The method is then iterated for the next left
    and right candidates (only one of them being different from before).

    This method ensures that the probability for a given candidate to be the peak is uniform
    (as opposed to the method :py:func:`~prefsampling.ordinal.single_peaked_walsh`). The
    probability for a single-peaked rank to be generated is equal to
    `1/m * (1/2)**dist_peak_to_end` where `m` is the number of candidates and `dist_peak_to_end`
    is the minimum distance from to peak to an end of the axis (i.e., candidates `0` or `m - 1`).

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

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
                votes[i][j:] = np.arange(right, num_candidates)
                break
            if right >= num_candidates:
                votes[i][j:] = np.arange(left, -1, -1)
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
    Left and right are defined here in terms of the circular axis: `0, 1, ..., m, 1`.

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

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
        left = np.mod(left, num_candidates)
        right = votes[j][0] + 1
        right = np.mod(right, num_candidates)
        for k in range(1, num_candidates):
            if rng.random() < 0.5:
                votes[j][k] = left
                left -= 1
                left = np.mod(left, num_candidates)
            else:
                votes[j][k] = right
                right += 1
                right = np.mod(right, num_candidates)
    return votes


@validate_num_voters_candidates
def single_peaked_walsh(
    num_voters: int, num_candidates: int, seed: int = None
) -> np.ndarray:
    """
    Generates ordinal votes that are single-peaked following the process described by
    `Walsh (2015) <https://arxiv.org/abs/1503.02766>`_. The votes are generated from least preferred
    to most preferred candidates. A given position in the ordering is filled by selecting, with
    uniform probability, either the leftmost or the rightmost candidates that have not yet been
    positioned in the vote (left and right being defined by the axis `0, 1, 2, ...`).

    A collection of `num_voters` vote is generated independently and identically following the
    process described above.

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
        position = num_candidates - 1
        left_most = 0
        right_most = num_candidates - 1
        while position != -1:
            if rng.random() < 0.5:
                votes[i, position] = left_most
                left_most += 1
            else:
                votes[i, position] = right_most
                right_most -= 1
            position -= 1

    return votes
