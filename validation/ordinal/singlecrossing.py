from prefsampling.ordinal import single_crossing
from prefsampling.ordinal.singlecrossing import single_crossing_impartial, \
    impartial_theoretical_distribution
from prefsampling.combinatorics import all_single_crossing_profiles, all_non_isomorphic_profiles, \
    all_anonymous_profiles
from validation.validator import Validator


class SingleCrossingValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 3, "num_candidates": 4},
        ]
        super(SingleCrossingValidator, self).__init__(
            parameters_list,
            "Single Crossing",
            "single_crossing",
            False,
            sampler_func=single_crossing,
            constant_parameters=("num_voters", "num_candidates"),
        )

    def all_outcomes(self, sampler_parameters):
        return all_single_crossing_profiles(
            sampler_parameters["num_voters"],
            sampler_parameters["num_candidates"],
            profiles=all_non_isomorphic_profiles(
                sampler_parameters["num_voters"],
                sampler_parameters["num_candidates"],
                profiles=all_anonymous_profiles(
                    sampler_parameters["num_voters"],
                    sampler_parameters["num_candidates"], ),
            ),
            fix_order=True,
        )

    def sample_cast(self, sample):
        return tuple(tuple(s) for s in sample)


class SingleCrossingImpartialValidator(SingleCrossingValidator):
    def __init__(self):
        super(SingleCrossingImpartialValidator, self).__init__()
        self.sampler_func = single_crossing_impartial
        self.model_name = "Single Crossing Impartial"
        self.model_short_name = "single_crossing_impartial"
        self.use_theoretical = True

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return impartial_theoretical_distribution(sc_profiles=all_outcomes)
