import numpy as np

from prefsampling.ordinal import group_separable
from validation.utils import get_all_group_separable_profiles, gs_structure
from validation.validator import Validator


class GroupSeparableValidator(Validator):
    def __init__(
        self,
        num_voters,
        num_candidates,
        tree,
        all_outcomes=None,
    ):
        super(GroupSeparableValidator, self).__init__(
            num_candidates,
            sampler_func=lambda num_samples, num_candidates, tree=None, num_voters=None: [
                group_separable(num_voters, num_candidates, tree=tree)
                for _ in range(num_samples)
            ],
            sampler_parameters={"num_voters": num_voters, "tree": tree},
            all_outcomes=all_outcomes,
        )
        self.num_voters = num_voters

    def set_all_outcomes(self):
        self.all_outcomes = get_all_group_separable_profiles(
            self.num_voters, self.num_candidates
        )

    def set_theoretical_distribution(self):
        self.theoretical_distribution = np.ones(len(self.all_outcomes)) / len(
            self.all_outcomes
        )

    def sample_cast(self, sample):
        return tuple(tuple(s) for s in sample)
