from prefsampling.combinatorics import all_rankings, all_anonymous_profiles
from prefsampling.ordinal import impartial, impartial_anonymous, stratification
from prefsampling.ordinal.impartial import (
    stratification_theoretical_distribution,
    impartial_theoretical_distribution,
    impartial_anonymous_theoretical_distribution,
)
from validation.validator import Validator


class OrdinalImpartialValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 4},
            {"num_voters": 1, "num_candidates": 5},
            {"num_voters": 1, "num_candidates": 6},
        ]
        super(OrdinalImpartialValidator, self).__init__(
            parameters_list,
            "Impartial",
            "impartial",
            True,
            sampler_func=impartial,
            constant_parameters="num_voters",
            faceted_parameters="num_candidates",
        )

    def all_outcomes(self, sampler_parameters):
        return all_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return impartial_theoretical_distribution(rankings=all_outcomes)

    def sample_cast(self, sample):
        return tuple(sample[0])


class OrdinalImpartialAnonymousValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 2, "num_candidates": 4},
            {"num_voters": 3, "num_candidates": 4},
        ]
        super(OrdinalImpartialAnonymousValidator, self).__init__(
            parameters_list,
            "Impartial Anonymous",
            "impartial_anonymous",
            True,
            sampler_func=impartial_anonymous,
            constant_parameters="num_candidates",
            faceted_parameters="num_voters",
        )

    def all_outcomes(self, sampler_parameters):
        return all_anonymous_profiles(
            sampler_parameters["num_voters"], sampler_parameters["num_candidates"]
        )

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return impartial_anonymous_theoretical_distribution(
            anonymous_profiles=all_outcomes
        )

    def sample_cast(self, sample):
        return tuple(sorted(tuple(s) for s in sample))


class OrdinalStratificationValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 5, "weight": 0.2},
            {"num_voters": 1, "num_candidates": 5, "weight": 0.4},
            {"num_voters": 1, "num_candidates": 5, "weight": 0.6},
        ]
        super(OrdinalStratificationValidator, self).__init__(
            parameters_list,
            "Stratification",
            "stratification",
            True,
            sampler_func=stratification,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters="weight",
        )

    def all_outcomes(self, sampler_parameters):
        return all_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return stratification_theoretical_distribution(
            sampler_parameters["num_candidates"],
            sampler_parameters["weight"],
            rankings=all_outcomes,
        )

    def sample_cast(self, sample):
        return tuple(sample[0])


class OrdinalStratificationUniformValidator(OrdinalStratificationValidator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 5, "weight": 0},
            {"num_voters": 1, "num_candidates": 5, "weight": 1},
        ]
        Validator.__init__(
            self,
            parameters_list,
            "Stratification",
            "stratification_uniform",
            True,
            sampler_func=stratification,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters="weight",
        )
