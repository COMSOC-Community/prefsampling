from prefsampling.approval import resampling, disjoint_resampling
from prefsampling.approval.resampling import (
    resampling_theoretical_distribution,
    moving_resampling,
    disjoint_resampling_theoretical_distribution,
)
from prefsampling.combinatorics import powerset
from validation.validator import Validator


class ApprovalResamplingValidator(Validator):
    def __init__(self):
        parameters_list = []
        for phi in [0.25, 0.5, 0.75, 1]:
            for rel_size_central_vote in [0.33, 0.5]:
                parameters_list.append(
                    {
                        "num_voters": 1,
                        "num_candidates": 6,
                        "phi": phi,
                        "rel_size_central_vote": rel_size_central_vote,
                        "central_vote": None,
                    }
                )
        super(ApprovalResamplingValidator, self).__init__(
            parameters_list,
            "Resampling",
            "resampling",
            True,
            sampler_func=resampling,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters=("phi", "rel_size_central_vote"),
        )

    def all_outcomes(self, sampler_parameters):
        return powerset(range(sampler_parameters["num_candidates"]))

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return resampling_theoretical_distribution(
            sampler_parameters["num_candidates"],
            sampler_parameters["phi"],
            sampler_parameters["rel_size_central_vote"],
            sampler_parameters["central_vote"],
        )

    def sample_cast(self, sample):
        return tuple(sorted(sample[0]))


class ApprovalDisjointResamplingValidator(Validator):
    def __init__(self):
        parameters_list = []
        for num_central_votes in [2, 3]:
            for rel_size_central_vote in [0.25, 0.33]:
                parameters_list.append(
                    {
                        "num_voters": 1,
                        "num_candidates": 6,
                        "phi": 0.5,
                        "rel_size_central_vote": rel_size_central_vote,
                        "num_central_votes": num_central_votes,
                    }
                )
        super(ApprovalDisjointResamplingValidator, self).__init__(
            parameters_list,
            "Disjoint Resampling",
            "disjoint_resampling",
            True,
            sampler_func=disjoint_resampling,
            constant_parameters=("num_voters", "num_candidates", "phi"),
            faceted_parameters=("rel_size_central_vote", "num_central_votes"),
        )

    def all_outcomes(self, sampler_parameters):
        return powerset(range(sampler_parameters["num_candidates"]))

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:

        return disjoint_resampling_theoretical_distribution(
            sampler_parameters["num_candidates"],
            sampler_parameters["phi"],
            sampler_parameters["rel_size_central_vote"],
            sampler_parameters["num_central_votes"],
            sampler_parameters["central_vote"],
        )

    def sample_cast(self, sample):
        return tuple(sorted(sample[0]))


class ApprovalMovingResamplingValidator(Validator):
    def __init__(self):
        parameters_list = []
        for phi in [0.25, 0.5, 0.75, 1]:
            for num_legs in [1, 2, 3]:
                parameters_list.append(
                    {
                        "num_voters": 4,
                        "num_candidates": 3,
                        "phi": phi,
                        "rel_size_central_vote": 0.33,
                        "num_legs": num_legs,
                        "central_vote": None,
                    }
                )
        super(ApprovalMovingResamplingValidator, self).__init__(
            parameters_list,
            "Moving Resampling",
            "moving_resampling",
            False,
            sampler_func=moving_resampling,
            constant_parameters=(
                "num_voters",
                "num_candidates",
                "rel_size_central_vote",
            ),
            faceted_parameters=("phi", "num_legs"),
        )

    def sample_cast(self, sample):
        return tuple(tuple(sorted(ballot)) for ballot in sample)
