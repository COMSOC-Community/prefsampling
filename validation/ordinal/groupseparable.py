from prefsampling.combinatorics import all_group_separable_profiles
from prefsampling.ordinal import group_separable, TreeSampler
from validation.validator import Validator


class GroupSeparableValidator(Validator):
    def __init__(self):
        parameters_list = [
            # {
            #     "num_voters": 3,
            #     "num_candidates": 3,
            #     "tree_sampler": TreeSampler.SCHROEDER,
            # },
            {
                "num_voters": 2,
                "num_candidates": 3,
                "tree_sampler": TreeSampler.SCHROEDER_UNIFORM,
            },
            # {
            #     "num_voters": 3,
            #     "num_candidates": 3,
            #     "tree_sampler": TreeSampler.SCHROEDER_LESCANNE,
            # },
            # {
            #     "num_voters": 3,
            #     "num_candidates": 3,
            #     "tree_sampler": TreeSampler.CATERPILLAR,
            # },
            # {
            #     "num_voters": 3,
            #     "num_candidates": 4,
            #     "tree_sampler": TreeSampler.SCHROEDER,
            # },
            # {
            #     "num_voters": 3,
            #     "num_candidates": 4,
            #     "tree_sampler": TreeSampler.SCHROEDER_UNIFORM,
            # },
            # {
            #     "num_voters": 3,
            #     "num_candidates": 4,
            #     "tree_sampler": TreeSampler.SCHROEDER_LESCANNE,
            # },
            # {
            #     "num_voters": 3,
            #     "num_candidates": 4,
            #     "tree_sampler": TreeSampler.CATERPILLAR,
            # },
        ]
        super(GroupSeparableValidator, self).__init__(
            parameters_list,
            "Group Separable",
            "group_separable",
            True,
            sampler_func=group_separable,
            constant_parameters="num_voters",
            faceted_parameters=("tree_sampler", "num_candidates"),
        )

    def all_outcomes(self, sampler_parameters):
        return all_group_separable_profiles(
            sampler_parameters["num_voters"],
            sampler_parameters["num_candidates"],
        )

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return {o: 1 / len(all_outcomes) for o in all_outcomes}

    def sample_cast(self, sample):
        return tuple(tuple(s) for s in sample)
