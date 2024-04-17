import numpy as np

from prefsampling.combinatorics import all_rankings
from prefsampling.ordinal import plackett_luce
from prefsampling.ordinal.plackettluce import theoretical_distribution
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
        return all_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return theoretical_distribution(sampler_parameters["alphas"], rankings=all_outcomes)

    def sample_cast(self, sample):
        return tuple(sample[0])
