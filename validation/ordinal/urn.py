import math

import numpy as np

from prefsampling.ordinal import urn
from validation.utils import get_all_anonymous_profiles
from validation.validator import Validator


class UrnValidator(Validator):
    def __init__(
        self,
        num_voters,
        num_candidates,
        alpha,
        all_outcomes=None,
    ):
        super(UrnValidator, self).__init__(
            num_candidates,
            sampler_func=lambda num_samples, num_candidates, alpha, num_voters=None: [
                urn(num_voters, num_candidates, alpha) for _ in range(num_samples)
            ],
            sampler_parameters={"num_voters": num_voters, "alpha": alpha},
            all_outcomes=all_outcomes,
        )
        self.num_voters = num_voters

    def set_all_outcomes(self):
        self.all_outcomes = get_all_anonymous_profiles(
            self.num_voters, self.num_candidates
        )

    def set_theoretical_distribution(self):
        def ascending_factorial(value, length, increment):
            if length == 0:
                return 1
            return (
                value + (length - 1) * increment * math.factorial(self.num_candidates)
            ) * ascending_factorial(value, length - 1, increment)

        distribution = np.zeros(len(self.all_outcomes))
        for i, profile in enumerate(self.all_outcomes):
            counts = {}
            for rank in profile:
                if rank in counts:
                    counts[rank] += 1
                else:
                    counts[rank] = 1
            probability = math.factorial(self.num_voters) / ascending_factorial(
                math.factorial(self.num_candidates),
                self.num_voters,
                self.sampler_parameters["alpha"],
            )
            for c in counts.values():
                probability *= ascending_factorial(
                    1, c, self.sampler_parameters["alpha"]
                ) / math.factorial(c)
            distribution[i] = probability
        self.theoretical_distribution = distribution / distribution.sum()

    def sample_cast(self, sample):
        return tuple(sorted(tuple(s) for s in sample))
