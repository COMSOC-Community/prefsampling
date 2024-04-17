from prefsampling.combinatorics import all_rankings
from prefsampling.ordinal import mallows
from prefsampling.ordinal.mallows import theoretical_distribution
from validation.validator import Validator


class OrdinalMallowsValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 5, "phi": 0.1, "normalise_phi": False},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.1, "normalise_phi": True},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.5, "normalise_phi": False},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.5, "normalise_phi": True},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.8, "normalise_phi": False},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.8, "normalise_phi": True},
            {"num_voters": 1, "num_candidates": 5, "phi": 1, "normalise_phi": False},
            {"num_voters": 1, "num_candidates": 5, "phi": 1, "normalise_phi": True},
        ]
        super(OrdinalMallowsValidator, self).__init__(
            parameters_list,
            "Mallows'",
            "mallows",
            True,
            sampler_func=mallows,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters=("phi", "normalise_phi"),
        )

    def all_outcomes(self, sampler_parameters):
        return all_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return theoretical_distribution(sampler_parameters["num_candidates"], sampler_parameters["phi"], normalise_phi=sampler_parameters["normalise_phi"], rankings=all_outcomes)

    def sample_cast(self, sample):
        return tuple(sample[0])
