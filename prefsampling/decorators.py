from functools import wraps


def validate_num_agents_candidates(func):
    """
    Decorator that tests that the first two arguments of the function are integers larger than 0.

    Parameters
    ----------
    func: Callable
        The decorated function

    """
    @wraps(func)
    def wrapper(num_agents, num_candidates, *args, **kwargs):
        try:
            assert int(num_agents) == num_agents
        except (ValueError, AssertionError):
            raise TypeError("The number of agents needs to be an integer.")
        if num_agents < 1:
            raise ValueError("The number of agents needs to be at least 1.")
        try:
            assert int(num_candidates) == num_candidates
        except (ValueError, AssertionError):
            raise TypeError("The number of candidates needs to be an integer.")
        if num_candidates < 1:
            raise ValueError("The number of candidates needs to be at least 1.")
        return func(num_agents, num_candidates, *args, **kwargs)
    return wrapper
