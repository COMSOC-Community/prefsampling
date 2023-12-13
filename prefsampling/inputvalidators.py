from functools import wraps


def validate_num_voters_candidates(func):
    """
    Decorator that tests that the first two arguments of the function are integers larger than 0.

    Parameters
    ----------
    func: Callable
        The decorated function

    """

    @wraps(func)
    def wrapper(num_voters, num_candidates, *args, **kwargs):
        try:
            assert int(num_voters) == num_voters
        except (ValueError, AssertionError):
            raise TypeError("The number of voters needs to be an integer.")
        if num_voters < 1:
            raise ValueError("The number of voters needs to be at least 1.")
        try:
            assert int(num_candidates) == num_candidates
        except (ValueError, AssertionError):
            raise TypeError("The number of candidates needs to be an integer.")
        if num_candidates < 1:
            raise ValueError("The number of candidates needs to be at least 1.")
        return func(num_voters, num_candidates, *args, **kwargs)

    return wrapper


def validate_positive_int(value, value_descr: str = ""):
    """
    Validates that the input value is an int greater or equal to 1.

    Parameters
    ----------
        value:
            The value to validate
        value_descr: str, default: ""
            A description of the value used in the message of the exceptions raised when the value is not a valid input.

    Raises
    ------
        TypeError
            When the value is not an int.
        ValueError
            When the value is 0 or less.
    """
    try:
        int(value)
    except TypeError:
        raise TypeError(f"The {value_descr} needs to be an integer.")
    if int(value) != value:
        raise TypeError(f"The {value_descr} needs to be an integer.")
    if value <= 0:
        raise ValueError(f"The {value_descr} needs to be 1 or more.")
