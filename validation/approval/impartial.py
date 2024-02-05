from prefsampling.approval import impartial
from validation.utils import get_all_subsets
from validation.validator import Validator
from math import factorial


class ApprovalImpartialValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 4, "p": 0.3},
            {"num_voters": 1, "num_candidates": 4, "p": 0.5},
            {"num_voters": 1, "num_candidates": 4, "p": 0.7},
            {"num_voters": 1, "num_candidates": 5, "p": 0.3},
            {"num_voters": 1, "num_candidates": 5, "p": 0.5},
            {"num_voters": 1, "num_candidates": 5, "p": 0.7},
        ]
        super(ApprovalImpartialValidator, self).__init__(
            parameters_list,
            "Impartial",
            "impartial",
            True,
            sampler_func=impartial,
            constant_parameters="num_voters",
            faceted_parameters=("p", "num_candidates"),
        )

    def all_outcomes(self, sampler_parameters):
        return get_all_subsets(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:

        m = sampler_parameters["num_candidates"]
        p = sampler_parameters["p"]

        probabilities = [None for _ in range(m+1)]
        for k in range(m+1):
            binomial_coefficient = factorial(m) / (factorial(k) * factorial(m - k))
            probability = binomial_coefficient * (p ** k) * ((1 - p) ** (m - k))
            probabilities[k] = probability

        return {str(len(o)): probabilities[len(o)] for o in all_outcomes}

    def sample_cast(self, sample):
        return str(len(sample[0]))
