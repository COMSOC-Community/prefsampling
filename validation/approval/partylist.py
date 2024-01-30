from prefsampling.approval import partylist
# from validation.utils import get_all_ranks, get_all_anonymous_profiles
from validation.validator import Validator


class ApprovalImpartialValidator(Validator):
    def __init__(self):
        parameters_list = [
            {"num_voters": 1, "num_candidates": 4},
            {"num_voters": 1, "num_candidates": 5},
            {"num_voters": 1, "num_candidates": 6},
        ]
        super(ApprovalImpartialValidator, self).__init__(
            parameters_list,
            "Partylist",
            "partylist",
            True,
            sampler_func=partylist,
            constant_parameters="num_voters",
            faceted_parameters="num_candidates",
        )

    # def all_outcomes(self, sampler_parameters):
    #     return get_all_ranks(sampler_parameters["num_candidates"])

    # def theoretical_distribution(self, sampler_parameters, all_outcomes) -> dict:
    #     return {o: 1 / len(all_outcomes) for o in all_outcomes}

    # def sample_cast(self, sample):
    #     return tuple(sample[0])
