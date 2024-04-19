from prefsampling.combinatorics import all_rankings
from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.ordinal import euclidean
from validation.validator import Validator


class OrdinalEuclideanValidatorUniform(Validator):
    def __init__(self):
        parameters_list = []
        for space in EuclideanSpace:
            for dimension in [2, 3]:
                parameters_list.append(
                    {
                        "num_voters": 1,
                        "num_candidates": 5,
                        "euclidean_space": space,
                        "num_dimensions": dimension,
                    },
                )
        super(OrdinalEuclideanValidatorUniform, self).__init__(
            parameters_list,
            "Euclidean",
            "euclidean_uniform",
            True,
            sampler_func=euclidean,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters=("euclidean_space", "num_dimensions"),
        )

    def sample_cast(self, sample):
        return tuple(sample[0])

    def all_outcomes(self, sampler_parameters):
        return all_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return {o: 1 / len(all_outcomes) for o in all_outcomes}


class OrdinalEuclideanValidator(Validator):
    def __init__(self):
        parameters_list = []
        for space in EuclideanSpace:
            for dimension in [2, 3]:
                parameters_list.append(
                    {
                        "num_voters": 3,
                        "num_candidates": 3,
                        "euclidean_space": space,
                        "num_dimensions": dimension,
                    },
                )
        super(OrdinalEuclideanValidator, self).__init__(
            parameters_list,
            "Euclidean",
            "euclidean",
            False,
            sampler_func=euclidean,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters=("euclidean_space", "num_dimensions"),
        )

    def sample_cast(self, sample):
        return tuple(tuple(r) for r in sample)
