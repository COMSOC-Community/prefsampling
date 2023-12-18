import numpy as np

from prefsampling.ordinal import (
    single_peaked_walsh,
    single_peaked_conitzer,
    single_peaked_circle,
)
from validation.utils import (
    get_all_single_peaked_ranks,
    get_all_single_peaked_circle_ranks,
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
        return get_all_single_peaked_ranks(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return {o: 1 / len(all_outcomes) for o in all_outcomes}

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
        return get_all_single_peaked_ranks(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        distribution = {}
        for rank in all_outcomes:
            probability = 1 / sampler_parameters["num_candidates"]
            for alt in rank:
                if alt == 0 or alt == sampler_parameters["num_candidates"] - 1:
                    break
                probability *= 1 / 2
            distribution[rank] = probability
        return distribution

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
        return get_all_single_peaked_circle_ranks(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return {o: 1 / len(all_outcomes) for o in all_outcomes}

    def sample_cast(self, sample):
        return tuple(sample[0])
