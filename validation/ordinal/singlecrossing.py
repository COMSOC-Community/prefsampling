from prefsampling.ordinal import single_crossing
from prefsampling.ordinal.singlecrossing import single_crossing_impartial
from validation.utils import get_all_sc_profiles_non_iso
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
        return get_all_sc_profiles_non_iso(
            sampler_parameters["num_voters"], sampler_parameters["num_candidates"]
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
        return {o: 1 / len(all_outcomes) for o in all_outcomes}
