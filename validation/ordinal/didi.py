import numpy as np

from prefsampling.ordinal import didi
from validation.validator import Validator


class DidiValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 5, "alphas": [0.1] * 5},
            {"num_voters": 1, "num_candidates": 5, "alphas": [1.0] + [0.3] * 4},
            {"num_voters": 1, "num_candidates": 5, "alphas": np.random.random(5)},
        ]
        super(DidiValidator, self).__init__(
            parameters_list,
            "Didi",
            "didi",
            False,
            sampler_func=didi,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters="alphas",
        )

    def sample_cast(self, sample):
        return tuple(sample[0])
