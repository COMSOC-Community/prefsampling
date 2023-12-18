from prefsampling.ordinal import mallows
from prefsampling.ordinal.mallows import phi_from_norm_phi
from validation.utils import get_all_ranks
from validation.validator import Validator


class OrdinalMallowsValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 5, "phi": 0.1, "normalise_phi": False},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.1, "normalise_phi": True},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.5, "normalise_phi": False},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.5, "normalise_phi": True},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.8, "normalise_phi": False},
            {"num_voters": 1, "num_candidates": 5, "phi": 0.8, "normalise_phi": True},
            {"num_voters": 1, "num_candidates": 5, "phi": 1, "normalise_phi": False},
            {"num_voters": 1, "num_candidates": 5, "phi": 1, "normalise_phi": True},
        ]
        super(OrdinalMallowsValidator, self).__init__(
            parameters_list,
            "Mallows'",
            "mallows",
            True,
            sampler_func=mallows,
            constant_parameters=("num_voters", "num_candidates"),
            faceted_parameters=("phi", "normalise_phi"),
        )

    def all_outcomes(self, sampler_parameters):
        return get_all_ranks(sampler_parameters["num_candidates"])

    def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
        distribution = {}
        if sampler_parameters["normalise_phi"]:
            phi = phi_from_norm_phi(
                sampler_parameters["num_candidates"], sampler_parameters["phi"]
            )
        else:
            phi = sampler_parameters["phi"]
        for rank in all_outcomes:
            distribution[rank] = phi ** kendall_tau_distance(
                tuple(range(sampler_parameters["num_candidates"])), rank
            )
        normaliser = sum(distribution.values())
        for r in distribution:
            distribution[r] /= normaliser
        return distribution

    def sample_cast(self, sample):
        return tuple(sample[0])


def kendall_tau_distance(rank1: tuple, rank2: tuple):
    distance = 0
    for k, alt1 in enumerate(rank1):
        for alt2 in rank1[k + 1 :]:
            if rank2.index(alt2) < rank2.index(alt1):
                distance += 1
    return distance
