from prefsampling.combinatorics import (
    all_single_peaked_rankings,
    all_single_peaked_circle_rankings,
)
from prefsampling.ordinal import (
    single_peaked_walsh,
    single_peaked_conitzer,
    single_peaked_circle,
)
from prefsampling.ordinal.singlepeaked import (
    walsh_theoretical_distribution,
    conitzer_theoretical_distribution,
    circle_theoretical_distribution,
)
from validation.validator import Validator


class SPWalshValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 4},
            {"num_voters": 1, "num_candidates": 5},
            {"num_voters": 1, "num_candidates": 6},
        ]
        super(SPWalshValidator, self).__init__(
            parameters_list,
            "Single-Peaked Walsh",
            "sp_walsh",
            True,
            sampler_func=single_peaked_walsh,
            constant_parameters="num_voters",
            faceted_parameters="num_candidates",
        )

    def all_outcomes(self, sampler_parameters):
        return all_single_peaked_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return walsh_theoretical_distribution(sp_rankings=all_outcomes)

    def sample_cast(self, sample):
        return tuple(sample[0])


class SPConitzerValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 4},
            {"num_voters": 1, "num_candidates": 5},
            {"num_voters": 1, "num_candidates": 6},
        ]
        super(SPConitzerValidator, self).__init__(
            parameters_list,
            "Single-Peaked Conitzer",
            "sp_conitzer",
            True,
            sampler_func=single_peaked_conitzer,
            constant_parameters="num_voters",
            faceted_parameters="num_candidates",
        )

    def all_outcomes(self, sampler_parameters):
        return all_single_peaked_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return conitzer_theoretical_distribution(
            sampler_parameters["num_candidates"], sp_rankings=all_outcomes
        )

    def sample_cast(self, sample):
        return tuple(sample[0])


class SPCircleValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 4},
            {"num_voters": 1, "num_candidates": 5},
            {"num_voters": 1, "num_candidates": 6},
        ]
        super(SPCircleValidator, self).__init__(
            parameters_list,
            "Single-Peaked on a Circle",
            "sp_circle",
            True,
            sampler_func=single_peaked_circle,
            constant_parameters="num_voters",
            faceted_parameters="num_candidates",
        )

    def all_outcomes(self, sampler_parameters):
        return all_single_peaked_circle_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return circle_theoretical_distribution(sp_circ_rankings=all_outcomes)

    def sample_cast(self, sample):
        return tuple(sample[0])
