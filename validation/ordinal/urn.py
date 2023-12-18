import math

from prefsampling.ordinal import urn
from validation.utils import get_all_anonymous_profiles
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
        return get_all_anonymous_profiles(
            sampler_parameters["num_voters"], sampler_parameters["num_candidates"]
        )

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        def ascending_factorial(value, length, increment):
            if length == 0:
                return 1
            return (
                value
                + (length - 1)
                * increment
                * math.factorial(sampler_parameters["num_candidates"])
            ) * ascending_factorial(value, length - 1, increment)

        distribution = {}
        for profile in all_outcomes:
            counts = {}
            for rank in profile:
                if rank in counts:
                    counts[rank] += 1
                else:
                    counts[rank] = 1
            probability = math.factorial(
                sampler_parameters["num_voters"]
            ) / ascending_factorial(
                math.factorial(sampler_parameters["num_candidates"]),
                sampler_parameters["num_voters"],
                sampler_parameters["alpha"],
            )
            for c in counts.values():
                probability *= ascending_factorial(
                    1, c, sampler_parameters["alpha"]
                ) / math.factorial(c)
            distribution[profile] = probability
        normaliser = sum(distribution.values())
        for r in distribution:
            distribution[r] /= normaliser
        return distribution

    def sample_cast(self, sample):
        return tuple(sorted(tuple(s) for s in sample))
