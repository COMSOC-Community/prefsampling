from prefsampling.ordinal import euclidean
from validation.utils import get_all_ranks
from validation.validator import Validator


class OrdinalEuclideanValidator(Validator):
    def __init__(self, num_candidates, space, dimension, all_outcomes=None):
        super(OrdinalEuclideanValidator, self).__init__(
            num_candidates,
            sampler_func=euclidean,
            sampler_parameters={"space": space, "dimension": dimension},
            all_outcomes=all_outcomes,
        )

    def set_all_outcomes(self):
        self.all_outcomes = get_all_ranks(self.num_candidates)

    def set_theoretical_distribution(self):
        pass

    def sample_cast(self, sample):
        return tuple(sample)
