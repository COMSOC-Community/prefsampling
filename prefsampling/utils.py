import math


def comb(n: int, k: int):
    """
    Function to compute the binomial coefficient. It uses math.comb if available (i.e., if Python >=
    3.8), otherwise computes it by hand.

    Parameters
    ----------
    n
    k

    Returns
    -------

    """
    if hasattr(math, "comb"):
        return math.comb(n, k)
    return _comb(n, k)


def _comb(n, k):
    try:
        return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
    except ValueError:
        return 0
