import numpy as np

from prefsampling.tree.schroeder import (
    schroeder_tree,
    all_schroeder_tree,
    _num_schroeder_tree,
)
from validation.validator import Validator


class SchroederValidator(Validator):
    def __init__(
        self,
        num_leaves,
        num_internal_nodes,
        sampler=schroeder_tree,
        all_outcomes=None,
    ):
        super(SchroederValidator, self).__init__(
            num_leaves,
            sampler_func=lambda num_samples, num_candidates, num_internal_nodes=None: [
                sampler(num_candidates, num_internal_nodes) for _ in range(num_samples)
            ],
            sampler_parameters={"num_internal_nodes": num_internal_nodes},
            all_outcomes=all_outcomes,
        )

    def set_all_outcomes(self):
        self.all_outcomes = [
            r.anonymous_tree_representation()
            for r in all_schroeder_tree(
                self.num_candidates, self.sampler_parameters["num_internal_nodes"]
            )
        ]

    def set_theoretical_distribution(self):
        self.theoretical_distribution = np.full(
            len(self.all_outcomes), 1 / len(self.all_outcomes)
        )

    def sample_cast(self, sample):
        return sample.anonymous_tree_representation()


class SchroederNumInternalValidator(Validator):
    def __init__(
        self,
        num_leaves,
        num_internal_nodes,
        sampler=schroeder_tree,
        all_outcomes=None,
    ):
        super(SchroederNumInternalValidator, self).__init__(
            num_leaves,
            sampler_func=lambda num_samples, num_candidates, num_internal_nodes=None: [
                sampler(num_candidates, num_internal_nodes) for _ in range(num_samples)
            ],
            sampler_parameters={"num_internal_nodes": num_internal_nodes},
            all_outcomes=all_outcomes,
        )

    def set_all_outcomes(self):
        self.all_outcomes = [
            r.anonymous_tree_representation()
            for r in all_schroeder_tree(
                self.num_candidates, self.sampler_parameters["num_internal_nodes"]
            )
        ]

    def set_theoretical_distribution(self):
        distribution = np.zeros(len(self.all_outcomes))
        for i, num_internal in enumerate(self.all_outcomes):
            distribution[i] = _num_schroeder_tree(num_internal, self.num_candidates)
        self.theoretical_distribution = distribution / distribution.sum()

    def sample_cast(self, sample):
        return sample.num_internal_nodes()
