from prefsampling.approval import impartial, impartial_constant_size
from prefsampling.combinatorics import powerset
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
        return powerset(range(sampler_parameters["num_candidates"]))

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        num_candidates = sampler_parameters["num_candidates"]
        p = sampler_parameters["p"]

        probabilities = []
        for k in range(num_candidates + 1):
            probabilities.append((p**k) * ((1 - p) ** (num_candidates - k)))

        return {tuple(sorted(o)): probabilities[len(o)] for o in all_outcomes}

    def sample_cast(self, sample):
        return tuple(sorted(sample[0]))


class ApprovalImpartialConstantSizeValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 5, "rel_num_approvals": 0.3},
            {"num_voters": 1, "num_candidates": 5, "rel_num_approvals": 0.5},
            {"num_voters": 1, "num_candidates": 5, "rel_num_approvals": 0.7},
            {"num_voters": 1, "num_candidates": 6, "rel_num_approvals": 0.3},
            {"num_voters": 1, "num_candidates": 6, "rel_num_approvals": 0.5},
            {"num_voters": 1, "num_candidates": 6, "rel_num_approvals": 0.7},
        ]
        super(ApprovalImpartialConstantSizeValidator, self).__init__(
            parameters_list,
            "Impartial Constant Size",
            "impartial_constant_size",
            True,
            sampler_func=impartial_constant_size,
            constant_parameters="num_voters",
            faceted_parameters=("rel_num_approvals", "num_candidates"),
        )

    def all_outcomes(self, sampler_parameters):
        return powerset(range(sampler_parameters["num_candidates"]))

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        num_candidates = sampler_parameters["num_candidates"]
        rel_num_approvals = sampler_parameters["rel_num_approvals"]

        size = int(rel_num_approvals * num_candidates)

        distribution = {o: int(len(o) == size) for o in all_outcomes}

        return {o: d / sum(distribution.values()) for o, d in distribution.items()}

    def sample_cast(self, sample):
        return tuple(sorted(sample[0]))
