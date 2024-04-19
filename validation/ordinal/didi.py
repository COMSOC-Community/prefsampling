from scipy import integrate

from prefsampling.combinatorics import all_rankings
from prefsampling.ordinal import didi
from validation.validator import Validator


class DidiValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 2, "alphas": [0.1, 0.1]},
            {"num_voters": 1, "num_candidates": 2, "alphas": [1.0, 0.3]},
            {"num_voters": 1, "num_candidates": 5, "alphas": [0.1] * 5},
            {"num_voters": 1, "num_candidates": 5, "alphas": [1.0] + [0.3] * 4},
            {"num_voters": 1, "num_candidates": 5, "alphas": [0.2, 0.5, 0.3, 0.7, 0.2]},
        ]
        super(DidiValidator, self).__init__(
            parameters_list,
            "DiDi",
            "didi",
            True,
            sampler_func=didi,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters="alphas",
        )

    def all_outcomes(self, sampler_parameters):
        return all_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        if min(sampler_parameters["alphas"]) == max(sampler_parameters["alphas"]):
            return {o: 1 / len(all_outcomes) for o in all_outcomes}
        if sampler_parameters["num_candidates"] == 2:
            alpha_0, alpha_1 = sampler_parameters["alphas"]
            prob_0_1 = integrate.quad(
                lambda x: x ** (alpha_0 - 1) * (1 - x) ** (alpha_1 - 1), 0.5, 1
            )[0]
            prob_1_0 = integrate.quad(
                lambda x: x ** (alpha_1 - 1) * (1 - x) ** (alpha_0 - 1), 0.5, 1
            )[0]
            denominator = prob_1_0 + prob_0_1
            return {(0, 1): prob_0_1 / denominator, (1, 0): prob_1_0 / denominator}

    def sample_cast(self, sample):
        return tuple(sample[0])
