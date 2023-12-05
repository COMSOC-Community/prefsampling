import numpy as np

from prefsampling.ordinal import plackett_luce
from validation.utils import get_all_ranks
from validation.validator import Validator


class PlackettLuceValidator(Validator):
    def __init__(
        self,
        num_candidates,
        alphas,
        all_outcomes=None,
    ):
        params = {"alphas": alphas}

        super(PlackettLuceValidator, self).__init__(
            num_candidates,
            sampler_func=plackett_luce,
            sampler_parameters=params,
            all_outcomes=all_outcomes,
        )

    def set_all_outcomes(self):
        self.all_outcomes = get_all_ranks(self.num_candidates)

    def set_theoretical_distribution(self):
        distribution = np.zeros(len(self.all_outcomes))
        norm_alphas = np.array(self.sampler_parameters["alphas"]) / sum(
            self.sampler_parameters["alphas"]
        )
        for i, rank in enumerate(self.all_outcomes):
            probability = 1
            for j, alt in enumerate(rank):
                probability *= norm_alphas[alt] / np.take(norm_alphas, rank[j:]).sum()
            distribution[i] = probability
        self.theoretical_distribution = distribution

    def sample_cast(self, sample):
        return tuple(sample)
