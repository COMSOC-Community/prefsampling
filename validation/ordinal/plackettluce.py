import numpy as np

from prefsampling.ordinal import plackett_luce
from validation.utils import get_all_ranks
from validation.validator import Validator


class PlackettLuceValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 5, "alphas": [0.1] * 5},
            {"num_voters": 1, "num_candidates": 5, "alphas": [1.0] + [0.3] * 4},
            {"num_voters": 1, "num_candidates": 5, "alphas": np.random.random(5)},
        ]
        super(PlackettLuceValidator, self).__init__(
            parameters_list,
            "Plackett-Luce",
            "plackett_luce",
            True,
            sampler_func=plackett_luce,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters="alphas",
        )

    def all_outcomes(self, sampler_parameters):
        return get_all_ranks(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        distribution = {}
        norm_alphas = np.array(sampler_parameters["alphas"]) / sum(
            sampler_parameters["alphas"]
        )
        for rank in all_outcomes:
            probability = 1
            for j, alt in enumerate(rank):
                probability *= norm_alphas[alt] / np.take(norm_alphas, rank[j:]).sum()
            distribution[rank] = probability
        normaliser = sum(distribution.values())
        for r in distribution:
            distribution[r] /= normaliser
        return distribution

    def sample_cast(self, sample):
        return tuple(sample[0])
