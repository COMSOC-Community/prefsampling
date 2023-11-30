import numpy as np

from prefsampling.ordinal import impartial
from validation.utils import get_all_ranks
from validation.validator import Validator


class OrdinalImpartialValidator(Validator):
    def __init__(self, num_candidates, all_outcomes=None):
        super(OrdinalImpartialValidator, self).__init__(
            num_candidates, sampler_func=impartial, all_outcomes=all_outcomes
        )

    def set_all_outcomes(self):
        self.all_outcomes = get_all_ranks(self.num_candidates)

    def set_theoretical_distribution(self):
        self.theoretical_distribution = np.full(
            len(self.all_outcomes), 1 / len(self.all_outcomes)
        )

    def sample_cast(self, sample):
        return tuple(sample)
