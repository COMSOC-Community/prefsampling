from prefsampling.approval import identity
from validation.utils import get_all_subsets
from validation.validator import Validator


class ApprovalIdentityValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 4, "p": 0.25},
            {"num_voters": 1, "num_candidates": 4, "p": 0.5},
        ]
        super(ApprovalIdentityValidator, self).__init__(
            parameters_list,
            "Identity",
            "identity",
            True,
            sampler_func=identity,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters="p",
        )

    def all_outcomes(self, sampler_parameters):
        return get_all_subsets(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        k = int(sampler_parameters["p"] * sampler_parameters["num_candidates"])
        return {str(k): 1}

    def sample_cast(self, sample):
        return str(len(sample[0]))
