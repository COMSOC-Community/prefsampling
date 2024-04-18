import math

from prefsampling.combinatorics import all_anonymous_profiles
from prefsampling.ordinal import urn
from prefsampling.ordinal.urn import theoretical_distribution
from validation.validator import Validator


class OrdinalUrnValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 3, "num_candidates": 4, "alpha": 0},
            {"num_voters": 3, "num_candidates": 4, "alpha": 1 / math.factorial(4)},
            {"num_voters": 3, "num_candidates": 4, "alpha": 0.5},
            {"num_voters": 3, "num_candidates": 4, "alpha": 1},
        ]
        super(OrdinalUrnValidator, self).__init__(
            parameters_list,
            "Urn",
            "urn",
            True,
            sampler_func=urn,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters="alpha",
        )

    def all_outcomes(self, sampler_parameters):
        return all_anonymous_profiles(
            sampler_parameters["num_voters"], sampler_parameters["num_candidates"]
        )

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return theoretical_distribution(sampler_parameters["num_voters"], sampler_parameters["num_candidates"], sampler_parameters["alpha"], profiles=all_outcomes)

    def sample_cast(self, sample):
        return tuple(sorted(tuple(s) for s in sample))
