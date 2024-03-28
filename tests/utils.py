import numpy as np

from prefsampling.core import (
    permute_voters,
    rename_candidates,
    resample_as_central_vote,
    mixture,
)


class TestSampler:
    def __init__(self, sampler, params, name=None):
        self.sampler = sampler
        self.params = params
        if name is None:
            name = f"{sampler.__name__}({params})"
        self.name = name
        self.test_method = None

    def test_sample_positional(self, num_voters, num_candidates, seed=None):
        return self.sampler(num_voters, num_candidates, seed=seed, **self.params)

    def test_sample_kwargs(self, num_voters, num_candidates, seed=None):
        return self.sampler(
            num_voters=num_voters,
            num_candidates=num_candidates,
            seed=seed,
            **self.params,
        )

    def test_sample(self, sample_method, num_voters, num_candidates):
        if sample_method == "positional":
            return self.test_sample_positional(num_voters, num_candidates)
        elif sample_method == "kwargs":
            return self.test_sample_kwargs(num_voters, num_candidates)
        elif sample_method == "seed":
            return self.test_sample_positional(num_voters, num_candidates, seed=3845)
        else:
            raise ValueError(
                "The 'sample_method' parameter needs to be one of: 'positional', "
                "'kwargs' or 'seed'."
            )

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def sample_then_permute(num_voters, num_candidates, main_test_sampler, seed=None):
    return permute_voters(
        main_test_sampler.test_sample_positional(num_voters, num_candidates, seed=seed),
        seed=seed,
    )


def sample_then_rename(num_voters, num_candidates, main_test_sampler, seed=None):
    return rename_candidates(
        main_test_sampler.test_sample_positional(num_voters, num_candidates, seed=seed),
        seed=seed,
    )


def sample_then_resample_as_central_vote(
    num_voters,
    num_candidates,
    main_test_sampler,
    resampler,
    resampler_params,
    seed=None,
):
    resampler_params["seed"] = seed
    resampler_params["num_candidates"] = num_candidates
    return resample_as_central_vote(
        main_test_sampler.test_sample_positional(num_voters, num_candidates, seed=seed),
        resampler,
        resampler_params,
    )


def sample_mixture(
    num_voters,
    num_candidates,
    test_sampler_1,
    test_sampler_2,
    test_sampler_3,
    seed=None,
):
    return mixture(
        num_voters,
        num_candidates,
        [
            test_sampler_1.test_sample_positional,
            test_sampler_2.test_sample_positional,
            test_sampler_3.test_sample_positional,
        ],
        [0.5, 0.2, 0.3],
        [{}, {}, {}],
    )


def int_parameter_test_values(lower_bound, upper_bound, num_samples):
    return [lower_bound, upper_bound] + list(
        np.random.randint(lower_bound + 1, upper_bound - 1, size=num_samples)
    )


def float_parameter_test_values(lower_bound, upper_bound, num_samples):
    values = [lower_bound, upper_bound]
    for _ in range(num_samples):
        v = np.random.random()
        v *= upper_bound - lower_bound
        v += lower_bound
        values.append(v)
    return values
