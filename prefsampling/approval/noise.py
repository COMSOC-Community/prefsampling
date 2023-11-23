import math

import numpy as np

from prefsampling.decorators import validate_num_voters_candidates


@validate_num_voters_candidates
def noise(
    num_voters: int = None,
    num_candidates: int = None,
    p: float = None,
    phi: float = 0.5,
    type_id: str = 'hamming',
    seed: int = None
) -> list[set]:
    """
     Generates approval votes from noise model.

     Parameters
     ----------
         num_voters : int
             Number of Voters.
         num_candidates : int
             Number of Candidates.
         phi : float, default: 0.5
             Noise model parameter, denoting the noise.
         p : float, default: 0.5
             Noise model parameter, denoting the length of central vote.
         type_id : str, default: hamming
             Type of noise.
             {'hamming', 'jaccard', 'zelinka', 'bunke-shearer'}
         seed : int
             Seed for numpy random number generator.

     Returns
     -------
         list[set]
             Approval votes.

     Raises
     ------
         ValueError
             When `phi` not in [0,1] interval.
             When `p` not in [0,1] interval.
             When `type_id` not in {'hamming', 'jaccard', 'zelinka', 'bunke-shearer'}.
     """

    if phi < 0 or 1 < phi:
        raise ValueError(f'Incorrect value of phi: {phi}. Value should be in [0,1]')

    if p < 0 or 1 < p:
        raise ValueError(f'Incorrect value of p: {p}. Value should be in [0,1]')

    if type_id not in {'hamming', 'jaccard', 'zelinka', 'bunke-shearer'}:
        raise ValueError(f'No such type_id as {type_id}')

    rng = np.random.default_rng(seed)

    k = int(p * num_candidates)

    A = {i for i in range(k)}
    B = set(range(num_candidates)) - A

    choices = []
    probabilities = []

    # Prepare buckets
    for x in range(len(A) + 1):
        num_options_in = math.comb(len(A), x)
        for y in range(len(B) + 1):
            num_options_out = math.comb(len(B), y)

            if type_id == 'hamming':
                factor = phi ** (len(A) - x + y)
            elif type_id == 'jaccard':
                factor = phi ** ((len(A) - x + y) / (len(A) + y))
            elif type_id == 'zelinka':
                factor = phi ** max(len(A) - x, y)
            elif type_id == 'bunke-shearer':
                factor = phi ** (max(len(A) - x, y) / max(len(A), x + y))
            else:
                factor = 1

            num_options = num_options_in * num_options_out * factor

            choices.append((x, y))
            probabilities.append(num_options)

    denominator = sum(probabilities)
    probabilities = [p / denominator for p in probabilities]

    # Sample Votes
    votes = []
    for _ in range(num_voters):
        _id = rng.choice(range(len(choices)), 1, p=probabilities)[0]
        x, y = choices[_id]
        vote = set(rng.choice(list(A), x, replace=False))
        vote = vote.union(set(rng.choice(list(B), y, replace=False)))
        votes.append(vote)

    return votes
