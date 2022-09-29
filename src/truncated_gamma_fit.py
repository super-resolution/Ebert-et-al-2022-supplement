import scipy
import numpy as np
import lmfit as lm


def pdf_gamma(x, alpha, beta):
    """
    Probability density function of the gamma distribution.

    Parameters
    ----------
    x : float, numpy.ndarray
        x-values of the distribution.
    alpha : float
        Shape parameter. Must be above 0.
    beta : float
        Rate parameter. Equal to 1/scale of scipy.stats.gamma

    Returns
    -------
    float or numpy.ndarray
        The probability density values for the given x.
    """
    return beta**alpha / scipy.special.gamma(alpha) * x**(alpha-1) * np.exp(-beta*x)


def cdf_gamma(x, alpha, beta):
    """
    Cumulative distribution function of the gamma distribution.

    Parameters
    ----------
    x : float, numpy.ndarray
        x-values of the distribution.
    alpha : float
        Shape parameter. Must be above 0.
    beta : float
        Rate parameter. Equal to 1/scale of scipy.stats.gamma

    Returns
    -------
    float or numpy.ndarray
        The probability of X ≤ x.
    """
    return scipy.special.gammainc(alpha, beta*x)


def pdf_truncated_gamma(x, alpha, beta, lower_bound, upper_bound=None):
    """
    Probability density function of the truncated gamma distribution.

    Parameters
    ----------
    x : float, numpy.ndarray
        x-values of the distribution.
    alpha : float
        Shape parameter. Must be above 0.
    beta : float
        Rate parameter. Equal to 1/scale of scipy.stats.gamma
    lower_bound : float
        Truncation point if lower x-values are missing.
    upper_bound : float
        Truncation point if upper x-values are missing.

    Returns
    -------
    float or numpy.ndarray
        The probability density values for the given x.
    """
    if upper_bound is None:
        return pdf_gamma(x, alpha, beta) / (1 - cdf_gamma(lower_bound, alpha, beta))
    else:
        return pdf_gamma(x, alpha, beta) / (cdf_gamma(upper_bound, alpha, beta) - cdf_gamma(lower_bound, alpha, beta))


def mean_trunc_gamma(alpha, beta, lower_bound, upper_bound=None):
    """
    Calculate the mean of the truncated gamma distribution given (a) boundary(ies).

    Parameters
    ----------
    alpha : float
        Shape parameter. Must be above 0.
    beta : float
        Rate parameter. Equal to 1/scale of scipy.stats.gamma
    lower_bound : float
        Truncation point if lower x-values are missing.
    upper_bound : float
        Truncation point if upper x-values are missing.

    Returns
    -------
    float
        The mean of the truncated gamma distribution.
    """
    if upper_bound:
        return scipy.integrate.quad(lambda x: x * pdf_gamma(x, alpha, beta),
                                    lower_bound, upper_bound)[0] / (cdf_gamma(upper_bound, alpha, beta) -
                                                                    cdf_gamma(lower_bound, alpha, beta))
    else:
        return scipy.integrate.quad(lambda x: x * pdf_gamma(x, alpha, beta),
                                    lower_bound, np.inf)[0] / (1 - cdf_gamma(lower_bound, alpha, beta))


def trunc_gamma_fit(y_data, x_data, alpha_values, beta_values, lower_bound, upper_bound=None):
    """
    Fitting the PDF of a truncated gamma distribution to data by minimization of the sum of squares.

    Parameters
    ----------
    y_data : numpy.ndarray
        Probability density values to fit, obtained e.g. by matplotlib.hist(density=True).
    x_data : numpy.ndarray
        Starting values of the bins corresponding to y_data, obtained e.g. by matplotlib.hist(density=True) excluding
        the last x-value.
    alpha_values : list
        Contains the initial, minimum and maximum value of the alpha parameter.
    beta_values : list
        Contains the initial, minimum and maximum value of the beta parameter.
    lower_bound : float
        Truncation point if lower x-values are missing.
    upper_bound : float, default=None
        Truncation point if upper x-values are missing.

    Returns
    -------
    alpha : float
        Expected alpha value of the (truncated) gamma distribution.
    beta : float
        Expected beta value of the (truncated) gamma distribution.
    """
    model = lm.Model(pdf_truncated_gamma, independent_vars=["x"])
    params = lm.Parameters()
    params.add("alpha", value=alpha_values[0], min=alpha_values[1], max=alpha_values[2])
    params.add("beta", value=beta_values[0], min=beta_values[1], max=beta_values[2])
    params.add("lower_bound", vary=False, value=lower_bound)
    if upper_bound:
        params.add("upper_bound", vary=False, value=upper_bound)

    result = model.fit(y_data, x=x_data, params=params)
    alpha = result.params["alpha"].value
    beta = result.params["beta"].value

    return alpha, beta
