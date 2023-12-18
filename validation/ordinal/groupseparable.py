from prefsampling.ordinal import group_separable, TreeSampler
from validation.validator import Validator


class GroupSeparableValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 3, "num_candidates": 3, "tree_sampler": TreeSampler.SCHROEDER},
            {"num_voters": 3, "num_candidates": 3, "tree_sampler": TreeSampler.SCHROEDER_UNIFORM},
            {"num_voters": 3, "num_candidates": 3, "tree_sampler": TreeSampler.SCHROEDER_LESCANNE},
            {"num_voters": 3, "num_candidates": 3, "tree_sampler": TreeSampler.CATERPILLAR},
            {"num_voters": 3, "num_candidates": 4, "tree_sampler": TreeSampler.SCHROEDER},
            {"num_voters": 3, "num_candidates": 4, "tree_sampler": TreeSampler.SCHROEDER_UNIFORM},
            {"num_voters": 3, "num_candidates": 4, "tree_sampler": TreeSampler.SCHROEDER_LESCANNE},
            {"num_voters": 3, "num_candidates": 4, "tree_sampler": TreeSampler.CATERPILLAR},
        ]
        super(GroupSeparableValidator, self).__init__(
            parameters_list,
            "Group Separable",
            "group_separable",
            False,
            sampler_func=group_separable,
            constant_parameters="num_voters",
            faceted_parameters=("tree_sampler", "num_candidates"),
        )

    def sample_cast(self, sample):
        return tuple(tuple(s) for s in sample)
