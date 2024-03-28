import numpy as np


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

