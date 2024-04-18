from prefsampling.approval import impartial
from prefsampling.combinatorics import powerset_as_sets
from validation.validator import Validator


class ApprovalImpartialValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 5, "p": 0.3},
            {"num_voters": 1, "num_candidates": 5, "p": 0.5},
            {"num_voters": 1, "num_candidates": 5, "p": 0.7},
            {"num_voters": 1, "num_candidates": 6, "p": 0.3},
            {"num_voters": 1, "num_candidates": 6, "p": 0.5},
            {"num_voters": 1, "num_candidates": 6, "p": 0.7},
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
        return powerset_as_sets(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        m = sampler_parameters["num_candidates"]
        p = sampler_parameters["p"]

        probabilities = []
        for k in range(m + 1):
            probabilities.append((p**k) * ((1 - p) ** (m - k)))

        return {tuple(sorted(o)): probabilities[len(o)] for o in all_outcomes}

    def sample_cast(self, sample):
        return tuple(sorted(sample[0]))
