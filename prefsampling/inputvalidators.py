from __future__ import annotations

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


def validate_int(
    value, value_descr: str = "value", lower_bound: int = None, upper_bound: int = None
):
    """
    Validates that the input value is an int. Lower and upper bounds on the value of the int can be
    provided.

    Parameters
    ----------
        value:
            The value to validate
        value_descr: str, default: :code:`"value"`
            A description of the value used in the message of the exceptions raised when the value
            is not a valid input.
        lower_bound: int, default: :code:`None`
            A lower bound on the value, the value cannot be strictly smaller than the bound.
        upper_bound: int, default: :code:`None`
            An upper bound on the value, the value cannot be strictly larger than the bound.

    Raises
    ------
        TypeError
            When the value is not an int or cannot be cast as an int.
        ValueError
            When the value is either strictly smaller than the lower bound or strictly greater than
            the upper bound.
    """
    try:
        int(value)
    except (ValueError, TypeError):
        raise TypeError(f"The {value_descr} needs to be an integer.")
    if int(value) != value:
        raise TypeError(f"The {value_descr} needs to be an integer.")
    if lower_bound is not None and value < lower_bound:
        raise ValueError(f"The {value_descr} needs to be {lower_bound} or more.")
    if upper_bound is not None and value > upper_bound:
        raise ValueError(f"The {value_descr} needs to be {upper_bound} or less.")
