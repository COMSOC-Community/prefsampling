import numpy as np

from prefsampling.ordinal import single_crossing
from validation.utils import get_all_single_crossing_profiles
from validation.validator import Validator


class SingleCrossingValidator(Validator):
    def __init__(
        self,
        num_voters,
        num_candidates,
        all_outcomes=None,
    ):
        super(SingleCrossingValidator, self).__init__(
            num_candidates,
            sampler_func=lambda num_samples, num_candidates, num_voters=None: [single_crossing(num_voters, num_candidates) for _ in range(num_samples)],
            sampler_parameters={"num_voters": num_voters},
            all_outcomes=all_outcomes,
        )
        self.num_voters = num_voters

    def set_all_outcomes(self):
        self.all_outcomes = get_all_single_crossing_profiles(self.num_voters, self.num_candidates)

    def set_theoretical_distribution(self):
        self.theoretical_distribution = np.ones(len(self.all_outcomes)) / len(self.all_outcomes)

    def sample_cast(self, sample):
        return tuple(tuple(s) for s in sample)
