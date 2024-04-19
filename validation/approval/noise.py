from prefsampling.approval import noise
from prefsampling.approval.noise import theoretical_distribution, SetDistance
from prefsampling.combinatorics import powerset
from validation.validator import Validator


class ApprovalNoiseValidator(Validator):
    def __init__(self):
        parameters_list = []
        for distance in SetDistance:
            for phi in [0, 0.25, 0.5, 0.75, 1]:
                parameters_list.append(
                    {
                        "num_voters": 1,
                        "num_candidates": 6,
                        "phi": phi,
                        "rel_size_central_vote": 0.5,
                        "distance": distance,
                        "central_vote": None,
                    }
                )
        super(ApprovalNoiseValidator, self).__init__(
            parameters_list,
            "Noise",
            "noise",
            True,
            sampler_func=noise,
            constant_parameters=(
                "num_voters",
                "num_candidates",
                "rel_size_central_vote",
            ),
            faceted_parameters=("phi", "distance"),
        )

    def all_outcomes(self, sampler_parameters):
        return powerset(range(sampler_parameters["num_candidates"]))

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return theoretical_distribution(
            sampler_parameters["num_candidates"],
            sampler_parameters["phi"],
            sampler_parameters["distance"],
            sampler_parameters["rel_size_central_vote"],
            sampler_parameters["central_vote"],
        )

    def sample_cast(self, sample):
        return tuple(sorted(sample[0]))
