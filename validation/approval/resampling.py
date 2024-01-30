import math

from prefsampling.approval import resampling
from validation.utils import get_all_subsets
from validation.validator import Validator


class ApprovalResamplingValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 6, "phi": 0.25, "p": 0.5},
            {"num_voters": 1, "num_candidates": 6, "phi": 0.5, "p": 0.5},
            {"num_voters": 1, "num_candidates": 6, "phi": 0.75, "p": 0.5},
            {"num_voters": 1, "num_candidates": 6, "phi": 0.25, "p": 0.34},
            {"num_voters": 1, "num_candidates": 6, "phi": 0.5, "p": 0.34},
            {"num_voters": 1, "num_candidates": 6, "phi": 0.75, "p": 0.34},
        ]
        super(ApprovalResamplingValidator, self).__init__(
            parameters_list,
            "Resampling",
            "resampling",
            True,
            sampler_func=resampling,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters=("phi", "p"),
        )

    def all_outcomes(self, sampler_parameters):
        return get_all_subsets(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        m = sampler_parameters["num_candidates"]
        p = sampler_parameters["p"]
        phi = sampler_parameters["phi"]
        k = math.floor(p*m)
        central_vote = {i for i in range(k)}

        A = {}
        for outcome in all_outcomes:
            prob = 1
            for c in range(m):
                if c in central_vote and c in outcome:
                    prob *= (1-phi) + phi*p
                elif c in central_vote and c not in outcome:
                    prob *= phi*(1-p)
                elif c not in central_vote and c in outcome:
                    prob *= phi*p
                else:
                    prob *= (1-phi) + phi*(1-p)
            A[str(outcome)] = prob
        return A

    def sample_cast(self, sample):
        return str(sample[0])

