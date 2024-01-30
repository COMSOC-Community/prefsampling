import math

from prefsampling.approval import noise
from validation.utils import get_all_subsets, hamming
from validation.validator import Validator


class ApprovalNoiseValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 4, "phi": 0.25, "p": 0.25},
            {"num_voters": 1, "num_candidates": 4, "phi": 0.25, "p": 0.5},
            {"num_voters": 1, "num_candidates": 4, "phi": 0.25, "p": 0.75},
            {"num_voters": 1, "num_candidates": 4, "phi": 0.5, "p": 0.25},
            {"num_voters": 1, "num_candidates": 4, "phi": 0.5, "p": 0.5},
            {"num_voters": 1, "num_candidates": 4, "phi": 0.5, "p": 0.75},
        ]
        super(ApprovalNoiseValidator, self).__init__(
            parameters_list,
            "Noise",
            "noise",
            True,
            sampler_func=noise,
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
        tmp_dict = {str(o): phi**hamming(central_vote, o) for o in all_outcomes}
        denom = sum(tmp_dict.values())
        return {str(o): tmp_dict[str(o)]/denom for o in all_outcomes}

    def sample_cast(self, sample):
        return str(sample[0])
