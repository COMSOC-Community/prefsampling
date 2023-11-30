import numpy as np

from prefsampling.ordinal import single_peaked_walsh, single_peaked_conitzer
from validation.utils import get_all_single_peaked_ranks
from validation.validator import Validator


class SPWalshValidator(Validator):
    def __init__(self, num_candidates, all_outcomes=None):
        super(SPWalshValidator, self).__init__(
            num_candidates,
            sampler_func=single_peaked_walsh,
            all_outcomes=all_outcomes,
        )

    def set_all_outcomes(self):
        self.all_outcomes = get_all_single_peaked_ranks(self.num_candidates)

    def set_theoretical_distribution(self):
        self.theoretical_distribution = np.full(
            len(self.all_outcomes), 1 / len(self.all_outcomes)
        )

    def sample_cast(self, sample):
        return tuple(sample)


class SPConitzerValidator(Validator):
    def __init__(self, num_candidates, all_outcomes=None):
        super(SPConitzerValidator, self).__init__(
            num_candidates,
            sampler_func=single_peaked_conitzer,
            all_outcomes=all_outcomes,
        )

    def set_all_outcomes(self):
        self.all_outcomes = get_all_single_peaked_ranks(self.num_candidates)

    def set_theoretical_distribution(self):
        distribution = np.zeros(len(self.all_outcomes))
        for i, rank in enumerate(self.all_outcomes):
            probability = 1 / self.num_candidates
            for alt in rank:
                if alt == 0 or alt == self.num_candidates - 1:
                    break
                probability *= 1 / 2
            distribution[i] = probability
        self.theoretical_distribution = distribution

    def sample_cast(self, sample):
        return tuple(sample)
