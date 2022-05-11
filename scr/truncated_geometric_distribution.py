import numpy as np
from scipy.stats import rv_discrete


def pmf_geometric(x, p, shift):
    """
    Probability mass function of the geometric distribution for x = {0, 1, 2, ...}.

    Parameters
    ----------
    x : int, numpy.ndarray
        x-values of the distribution.
    p : float
        Success probability. 0 < p ≤ 1
    shift : int
        To shift the distribution in positive x-direction, use value > 0.

    Returns
    -------
    float or numpy.ndarray
        The probability mass values for the given x.
    """
    return (1 - p)**(x-shift) * p


def cdf_geometric(x, p, shift):
    """
    Cumulative distribution function of the geometric distribution.

    Parameters
    ----------
    x : int, numpy.ndarray
        x-values of the distribution.
    p : float
        Success probability. 0 < p ≤ 1
    shift : int
        To shift the distribution in positive x-direction, use value > 0.

    Returns
    -------
    float or numpy.ndarray
        The probability of X ≤ x.
    """
    return 1 - (1 - p) ** (x + 1 - shift)


def pmf_truncated_geometric(x, p, upper_bound, lower_bound, shift):
    """
    Probability mass function of the truncated geometric distribution.

    Parameters
    ----------
    x : int, numpy.ndarray
        x-values of the distribution.
    p : float
        Success probability. 0 < p ≤ 1
    upper_bound : int
        Truncation point if lower x-values are missing.
    lower_bound : int
        Truncation point if upper x-values are missing.
    shift : int
        To shift the distribution in positive x-direction, use value > 0.

    Returns
    -------
    float or numpy.ndarray
        The probability mass values for the given x.
    """
    return pmf_geometric(x, p, shift) / (cdf_geometric(upper_bound, p, shift) - cdf_geometric(lower_bound, p, shift))


def truncated_geometric(p, upper_bound, lower_bound, shift):
    """
    Construct an arbitrary distribution defined by a list of support points and their corresponding probabilities.
    In this case, the support points range between and including lower_bound + 1 and upper_bound. Their probabilities
    are estimated with pmf_truncated_geometric.

    Parameters
    ----------
    p : float
        Success probability. 0 < p ≤ 1
    upper_bound : int
        Truncation point if lower x-values are missing.
    lower_bound : int
        Truncation point if upper x-values are missing.
    shift : int
        To shift the distribution in positive x-direction, use value > 0.

    Returns
    -------

    """
    x = np.arange(lower_bound+1, upper_bound+1, 1)  # the support of truncated pmf is half-open interval (a, b]
    probabilities = pmf_truncated_geometric(x, p, upper_bound, lower_bound, shift)

    trunc_geo = rv_discrete(name="trunc_geo", values=(x, probabilities))

    return trunc_geo


def mean_trunc_geo(p, lower_bound, upper_bound, shift):
    """
    Calculate the mean of the truncated geometric distribution given (a) boundary(ies).

    Parameters
    ----------
    p : float
        Success probability. 0 < p ≤ 1
    upper_bound : int
        Truncation point if lower x-values are missing.
    lower_bound : int
        Truncation point if upper x-values are missing.
    shift : int
        To shift the distribution in positive x-direction, use value > 0.

    Returns
    -------
    float
        The mean of the truncated geometric distribution.
    """
    x = np.arange(lower_bound+1, upper_bound+1, 1)
    return np.sum(x * pmf_geometric(x, p, shift)) / (cdf_geometric(upper_bound, p, shift) -
                                                     cdf_geometric(lower_bound, p, shift))
