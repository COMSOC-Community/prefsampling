from prefsampling.tree.schroeder import (
    schroeder_tree,
    schroeder_tree_lescanne,
    schroeder_tree_brute_force,
)
from validation.validator import Validator


class SchroederValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_leaves": 4, "num_internal_nodes": None},
            {"num_leaves": 4, "num_internal_nodes": 1},
            {"num_leaves": 4, "num_internal_nodes": 2},
            {"num_leaves": 4, "num_internal_nodes": 3},
            {"num_leaves": 5, "num_internal_nodes": None},
            {"num_leaves": 5, "num_internal_nodes": 1},
            {"num_leaves": 5, "num_internal_nodes": 2},
            {"num_leaves": 5, "num_internal_nodes": 3},
            {"num_leaves": 5, "num_internal_nodes": 4},
        ]
        super(SchroederValidator, self).__init__(
            parameters_list,
            "Schröder tree",
            "schroeder",
            False,
            sampler_func=schroeder_tree,
            faceted_parameters=("num_internal_nodes", "num_leaves"),
        )

    def sample_cast(self, sample):
        return sample.anonymous_tree_representation()


class SchroederLescanneValidator(SchroederValidator):
    def __init__(self):
        super().__init__()
        self.sampler_func = schroeder_tree_lescanne
        self.model_name = "Schröder tree by Lescanne (2022)"
        self.model_short_name = "schroeder_lescanne"


class SchroederBruteForceValidator(SchroederValidator):
    def __init__(self):
        super().__init__()
        self.sampler_func = schroeder_tree_brute_force
        self.model_name = "Schröder tree by brute force"
        self.model_short_name = "schroeder_brute_force"

    def all_outcomes(self, sampler_parameters):
        pass
