import math

import numpy as np

from prefsampling.ordinal import mallows, norm_mallows
from validation.utils import get_all_ranks
from validation.validator import Validator


class OrdinalMallowsValidator(Validator):
    def __init__(
        self,
        num_candidates,
        phi,
        central_vote,
        use_norm_mallows=False,
        all_outcomes=None,
    ):
        params = {"phi": phi, "central_vote": central_vote}
        if use_norm_mallows:
            sampler = norm_mallows
        else:
            sampler = mallows
        super(OrdinalMallowsValidator, self).__init__(
            num_candidates,
            sampler_func=sampler,
            sampler_parameters=params,
            all_outcomes=all_outcomes,
        )

    def set_all_outcomes(self):
        self.all_outcomes = get_all_ranks(self.num_candidates)

    def set_theoretical_distribution(self):
        distribution = np.zeros(len(self.all_outcomes))
        for i, rank in enumerate(self.all_outcomes):
            distribution[i] = self.sampler_parameters["phi"] ** kendall_tau_distance(
                self.sampler_parameters["central_vote"], rank
            )
        self.theoretical_distribution = distribution / sum(distribution)

    def sample_cast(self, sample):
        return tuple(sample)


def kendall_tau_distance(rank1: tuple, rank2: tuple):
    distance = 0
    for k, alt1 in enumerate(rank1):
        for alt2 in rank1[k + 1 :]:
            if rank2.index(alt2) < rank2.index(alt1):
                distance += 1
    return distance


def frequencies_by_distance(
    frequencies: np.ndarray, central_rank: tuple, all_ranks: list[tuple[int]]
):
    result = np.zeros(math.comb(len(all_ranks[0]), 2) + 1)
    for i, freq in enumerate(frequencies):
        result[kendall_tau_distance(all_ranks[i], central_rank)] += freq
    return result
