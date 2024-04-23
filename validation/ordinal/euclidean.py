import numpy as np
import scipy

from prefsampling.combinatorics import all_rankings
from prefsampling.core.euclidean import EuclideanSpace
from prefsampling.ordinal import euclidean
from prefsampling.point import ball_uniform
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
                        "voters_positions": space,
                        "candidates_positions": space,
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
            faceted_parameters=("voters_positions", "num_dimensions"),
        )

    def sample_cast(self, sample):
        return tuple(sample[0])

    def all_outcomes(self, sampler_parameters):
        return all_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        return {o: 1 / len(all_outcomes) for o in all_outcomes}


class OrdinalEuclideanValidatorLine(Validator):
    def __init__(self):
        parameters_list = []
        spaces = (
            (EuclideanSpace.UNIFORM_BALL, {"widths": 1, "center_point": [0.5]}),
            (EuclideanSpace.UNIFORM_CUBE, {"widths": 1, "center_point": [0.5]}),
            (EuclideanSpace.UNBOUNDED_GAUSSIAN, {"sigmas": 0.33, "center_point": [0.5]})
        )
        for space, space_params in spaces:
            for candidates_positions in [[0.1, 0.3, 0.75, 0.78], [0.1, 0.12, 0.8]]:
                parameters_list.append(
                    {
                        "num_voters": 1,
                        "num_candidates": len(candidates_positions),
                        "voters_positions": space,
                        "voters_positions_args": space_params,
                        "candidates_positions": candidates_positions,
                        "num_dimensions": 1,
                    },
                )
        super(OrdinalEuclideanValidatorLine, self).__init__(
            parameters_list,
            "Euclidean",
            "euclidean_line",
            True,
            sampler_func=euclidean,
            constant_parameters=("num_voters"),
            faceted_parameters=("voters_positions", "candidates_positions"),
        )

    def sample_cast(self, sample):
        return tuple(sample[0])

    def all_outcomes(self, sampler_parameters):
        return all_rankings(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        num_candidates = sampler_parameters["num_candidates"]
        candidates_positions = sampler_parameters["candidates_positions"]
        space = sampler_parameters["voters_positions"]
        transition_positions = []
        for j in range(num_candidates):
            pos_j = candidates_positions[j]
            for k in range(j + 1, num_candidates):
                pos_k = candidates_positions[k]
                if pos_j < pos_k:
                    trans_pos = pos_j + (pos_k - pos_j) / 2
                    trans = (k, j)
                else:
                    trans_pos = pos_k + (pos_j - pos_k) / 2
                    trans = (j, k)
                transition_positions.append((trans_pos, trans))

        transition_positions.sort(key=lambda x: x[0])

        distribution = {o: 0 for o in all_outcomes}
        current_order = tuple(np.argsort(np.array(candidates_positions)))
        current_pos = -float("inf") if space == EuclideanSpace.UNBOUNDED_GAUSSIAN else 0
        for trans_pos, trans in transition_positions:
            if space == EuclideanSpace.UNBOUNDED_GAUSSIAN:
                distribution[current_order] = scipy.stats.norm(0.5, 0.33).cdf(trans_pos) - scipy.stats.norm(0.5, 0.33).cdf(current_pos)
            else:
                distribution[current_order] = trans_pos - current_pos
            new_order = list(current_order)
            new_order[current_order.index(trans[0])] = trans[1]
            new_order[current_order.index(trans[1])] = trans[0]
            current_order = tuple(new_order)
            current_pos = trans_pos
        if space == EuclideanSpace.UNBOUNDED_GAUSSIAN:
            distribution[current_order] = scipy.stats.norm(0.5, 0.33).cdf(float("inf")) - scipy.stats.norm(0.5, 0.33).cdf(current_pos)
        else:
            distribution[current_order] = 1 - current_pos
        return {o: d / sum(distribution.values()) for o, d in distribution.items()}


class OrdinalEuclideanValidator(Validator):
    def __init__(self):
        parameters_list = []
        for space in EuclideanSpace:
            for dimension in [2, 3]:
                parameters_list.append(
                    {
                        "num_voters": 3,
                        "num_candidates": 3,
                        "voters_positions": space,
                        "candidates_positions": space,
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
            faceted_parameters=("voters_positions", "num_dimensions"),
        )

    def sample_cast(self, sample):
        return tuple(tuple(r) for r in sample)
