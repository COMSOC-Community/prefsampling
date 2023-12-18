from prefsampling.ordinal import impartial, impartial_anonymous, stratification
from validation.utils import get_all_ranks, get_all_anonymous_profiles
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
        return get_all_ranks(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return {o: 1 / len(all_outcomes) for o in all_outcomes}

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
        return get_all_anonymous_profiles(
            sampler_parameters["num_voters"], sampler_parameters["num_candidates"]
        )

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return {o: 1 / len(all_outcomes) for o in all_outcomes}

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
        return get_all_ranks(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        upper_class_size = int(
            sampler_parameters["weight"] * sampler_parameters["num_candidates"]
        )
        upper_class_candidates = set(range(upper_class_size))
        distribution = {}
        for rank in all_outcomes:
            if set(rank[:upper_class_size]) == upper_class_candidates:
                distribution[rank] = 1
            else:
                distribution[rank] = 0
        normaliser = sum(distribution.values())
        for r in distribution:
            distribution[r] /= normaliser
        return distribution

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
