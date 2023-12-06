import numpy as np

from prefsampling.ordinal import impartial, impartial_anonymous, stratification
from validation.utils import get_all_ranks, get_all_profiles
from validation.validator import Validator


class OrdinalImpartialValidator(Validator):
    def __init__(self, num_candidates, all_outcomes=None):
        super(OrdinalImpartialValidator, self).__init__(
            num_candidates,
            sampler_func=impartial,
            all_outcomes=all_outcomes,
        )

    def set_all_outcomes(self):
        self.all_outcomes = get_all_ranks(self.num_candidates)

    def set_theoretical_distribution(self):
        self.theoretical_distribution = np.full(
            len(self.all_outcomes), 1 / len(self.all_outcomes)
        )

    def sample_cast(self, sample):
        return tuple(sample)


class OrdinalImpartialAnonymousValidator(Validator):
    def __init__(
        self,
        num_voters,
        num_candidates,
        all_outcomes=None,
    ):
        super(OrdinalImpartialAnonymousValidator, self).__init__(
            num_candidates,
            sampler_func=lambda num_samples, num_candidates, num_voters=None: [
                impartial_anonymous(num_voters, num_candidates)
                for _ in range(num_samples)
            ],
            sampler_parameters={"num_voters": num_voters},
            all_outcomes=all_outcomes,
        )
        self.num_voters = num_voters

    def set_all_outcomes(self):
        self.all_outcomes = get_all_profiles(self.num_voters, self.num_candidates)

    def set_theoretical_distribution(self):
        self.theoretical_distribution = np.full(
            len(self.all_outcomes), 1 / len(self.all_outcomes)
        )

    def sample_cast(self, sample):
        return tuple(sorted(tuple(s) for s in sample))


class StratificationValidator(Validator):
    def __init__(self, num_candidates, weight, all_outcomes=None):
        super(StratificationValidator, self).__init__(
            num_candidates,
            sampler_func=stratification,
            all_outcomes=all_outcomes,
            sampler_parameters={"weight": weight},
        )

    def set_all_outcomes(self):
        self.all_outcomes = get_all_ranks(self.num_candidates)

    def set_theoretical_distribution(self):
        upper_class_size = int(self.sampler_parameters["weight"] * self.num_candidates)
        upper_class_candidates = set(range(upper_class_size))
        distribution = np.zeros(len(self.all_outcomes))
        for i, rank in enumerate(self.all_outcomes):
            if set(rank[:upper_class_size]) == upper_class_candidates:
                distribution[i] = 1
        self.theoretical_distribution = distribution / sum(distribution)

    def sample_cast(self, sample):
        return tuple(sample)
