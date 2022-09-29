import ray
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.special as special
import scipy.spatial as spatial


def pdf_normal(x):
    """
    Probability density function of the normal distribution.
    """
    return 1 / (np.sqrt(2 * np.pi)) * np.exp((-1 / 2) * (x ** 2))


def cdf_normal(x):
    """
    Cumulative distribution function of the normal distribution.
    """
    return 1 / 2 * (1 + special.erf(x / np.sqrt(2)))


# 2D normal distributed points - perimeter of convex hull


def sim_peri_convex_hull_2d(n_max_points, seed, n_simulations, sigma):
    """
    Simulate n normal distributed points in 2D and calculate their perimeter of the convex hull.

    Parameters
    ----------
    n_max_points : int
        Maximum amount of points.
    seed : int, array_like, numpy.random.BitGenerator
        Seed of random number generation.
    n_simulations : int
        Amount of simulations.
    sigma : float
        Standard deviation of the normal distribution.

    Returns
    -------
    array_sim_peri : numpy.ndarray
        Array of shape ((n_max_points - n_points_start), n_simulations).
    """
    rng = np.random.default_rng(seed)
    array_sim_peri = np.empty((0, n_simulations))
    n_points = 2
    for i in range(n_max_points - 2):
        n_points += 1
        points = rng.normal(loc=0, scale=sigma, size=(n_simulations, n_points, 2))
        perimeter = np.array([spatial.ConvexHull(pts).area for pts in points])
        perimeter = perimeter.reshape(1, n_simulations)
        array_sim_peri = np.append(array_sim_peri, perimeter, 0)

    return array_sim_peri


def calc_peri_convex_hull_2d_pt1(n):
    """
    Calculate the expected value of the perimeter of the convex hull of n normal distributed points in 2D.
    The standard deviation is 1.

    Parameters
    ----------
    n : int
        Amount of normal distributed points.

    Returns
    -------
    float
    """
    return 4 * np.pi * special.binom(n, 2) * integrate.quad(
        lambda x: cdf_normal(x) ** (n - 2) * pdf_normal(x) ** 2, -np.inf, np.inf)[0]


def calc_peri_convex_hull_2d_pt2(sigma, n):
    """
    Calculate the expected value of the perimeter of the convex hull of n normal distributed points in 2D.

    Parameters
    ----------
    sigma : float
        Standard deviation of the normal distribution.
    n : int, array_like
        Amount of normal distributed points.

    Returns
    -------
    float or array
    """
    calc_peri_convex_hull_2d_pt1_array = np.vectorize(calc_peri_convex_hull_2d_pt1)

    return sigma * calc_peri_convex_hull_2d_pt1_array(n)


# 2D normal distributed points - area of convex hull


def sim_area_convex_hull_2d(n_max_points, seed, n_simulations, sigma):
    """
    Simulate n normal distributed points in 2D and calculate their area of the convex hull.

    Parameters
    ----------
    n_max_points : int
        Maximum amount of points.
    seed : int, array_like, numpy.random.BitGenerator
        Seed of random number generation.
    n_simulations : int
        Amount of simulations.
    sigma : float
        Standard deviation of the normal distribution.

    Returns
    -------
    array_sim_areas : numpy.ndarray
        Array of shape ((n_max_points - n_points_start), n_simulations).
    """
    rng = np.random.default_rng(seed)
    array_sim_areas = np.empty((0, n_simulations))
    n_points = 2
    for i in range(n_max_points - 2):
        n_points += 1
        points = rng.normal(loc=0, scale=sigma, size=(n_simulations, n_points, 2))
        area = np.array([spatial.ConvexHull(pts).volume for pts in points])
        area = area.reshape(1, n_simulations)
        array_sim_areas = np.append(array_sim_areas, area, 0)

    return array_sim_areas


def calc_area_convex_hull_2d_pt1(n):
    """
    Calculate the expected value of the area of the convex hull of n normal distributed points in 2D.
    The standard deviation is 1.

    Parameters
    ----------
    n : int
        Amount of normal distributed points.

    Returns
    -------
    float
    """
    return 3 * np.pi * special.binom(n, 3) * integrate.quad(
        lambda x: cdf_normal(x) ** (n - 3) * pdf_normal(x) ** 3, -np.inf, np.inf)[0]


def calc_area_convex_hull_2d_pt2(sigma, n):
    """
    Calculate the expected value of the area of the convex hull of n normal distributed points in 2D.

    Parameters
    ----------
    sigma : float
        Standard deviation of the normal distribution.
    n : int, array_like
        Amount of normal distributed points.

    Returns
    -------
    float or array
    """
    calc_area_convex_hull_2d_pt1_array = np.vectorize(calc_area_convex_hull_2d_pt1)

    return sigma ** 2 * calc_area_convex_hull_2d_pt1_array(n)


# 3D normal distributed points - surface area of convex hull


def sim_area_convex_hull_3d(n_max_points, seed, n_simulations, sigma_xy, sigma_z):
    """
    Simulate n normal distributed points in 3D and calculate their surface area of the convex hull.

    Parameters
    ----------
    n_max_points : int
        Maximum amount of points.
    seed : int, array_like, numpy.random.BitGenerator
        Seed of random number generation.
    n_simulations : int
        Amount of simulations.
    sigma_xy : float
        Standard deviation of the normal distribution in x,y plane.
    sigma_z : float
        Standard deviation of the normal distribution in z direction.

    Returns
    -------
    array_sim_areas : numpy.ndarray
        Array of shape ((n_max_points - n_points_start), n_simulations).
    """
    rng = np.random.default_rng(seed)
    array_sim_areas = np.empty((0, n_simulations))
    n_points = 3
    for i in range(n_max_points - 3):
        n_points += 1
        points = rng.normal(loc=0, scale=sigma_xy, size=(n_simulations, n_points, 2))
        z_coordinate = rng.normal(loc=0, scale=sigma_z, size=(n_simulations, n_points, 1))
        points = np.append(points, z_coordinate, axis=2)
        area = np.array([spatial.ConvexHull(pts).area for pts in points])
        area = area.reshape(1, n_simulations)
        array_sim_areas = np.append(array_sim_areas, area, 0)

    return array_sim_areas


def calc_area_convex_hull_3d_pt1(n):
    """
    Calculate the expected value of the surface area of the convex hull of n normal distributed points in 3D.
    The standard deviation is 1.

    Parameters
    ----------
    n : int
        Amount of normal distributed points.

    Returns
    -------
    float
    """
    return 12 * np.pi * special.binom(n, 3) * integrate.quad(
        lambda x: cdf_normal(x) ** (n - 3) * pdf_normal(x) ** 3, -np.inf, np.inf)[0]


def calc_factor(sigma_xy, sigma_z):
    """
    Calculate the factor with which calc_area_convex_hull_3d_pt1 has to be multiplied with to be adjusted to
    standard deviations sigma_xy, sigma_z that are not equal and not 1.

    Parameters
    ----------
    sigma_xy : float
        Standard deviation of the normal distribution in x,y plane.
    sigma_z : float
        Standard deviation of the normal distribution in z direction.
    Returns
    -------
    float
    """
    if sigma_xy == sigma_z:
        factor = sigma_xy ** 2
    elif sigma_xy < sigma_z:
        e = np.sqrt(1 - (sigma_xy ** 2 / sigma_z ** 2))
        a = 2 * np.pi * sigma_xy ** 2 * (1 + (sigma_z / (sigma_xy * e)) * np.arcsin(e))
        factor = a / (4 * np.pi)
    elif sigma_xy > sigma_z:
        e = np.sqrt(1 - (sigma_z ** 2 / sigma_xy ** 2))
        a = 2 * np.pi * sigma_xy ** 2 + np.pi * (sigma_z ** 2 / e) * np.log((1 + e) / (1 - e))
        factor = a / (4 * np.pi)
    else:
        factor = 1

    return factor


def calc_area_convex_hull_3d_pt2(sigma_xy, sigma_z, n):
    """
    Calculate the expected value of the surface area of the convex hull of n normal distributed points in 3D.

    Parameters
    ----------
    sigma_xy : float
        Standard deviation of the normal distribution in x,y plane.
    sigma_z : float
        Standard deviation of the normal distribution in z direction.
    n : int, array_like
        Amount of normal distributed points.

    Returns
    -------
    float or array
    """
    calc_area_convex_hull_3d_pt1_array = np.vectorize(calc_area_convex_hull_3d_pt1)

    return calc_factor(sigma_xy, sigma_z) * calc_area_convex_hull_3d_pt1_array(n)


# 3D normal distributed points - volume of convex hull


def sim_volume_convex_hull_3d(n_max_points, seed, n_simulations, sigma_xy, sigma_z):
    """
    Simulate n normal distributed points in 3D and calculate their volume of the convex hull.

    Parameters
    ----------
    n_max_points : int
        Maximum amount of points.
    seed : int, array_like, numpy.random.BitGenerator
        Seed of random number generation.
    n_simulations : int
        Amount of simulations.
    sigma_xy : float
        Standard deviation of the normal distribution in x,y plane.
    sigma_z : float
        Standard deviation of the normal distribution in z direction.

    Returns
    -------
    array_sim_areas : numpy.ndarray
        Array of shape ((n_max_points - n_points_start), n_simulations).
    """
    rng = np.random.default_rng(seed)
    array_sim_volume = np.empty((0, n_simulations))
    n_points = 3
    for i in range(n_max_points - 3):
        n_points += 1
        points = rng.normal(loc=0, scale=sigma_xy, size=(n_simulations, n_points, 2))
        z_coordinate = rng.normal(loc=0, scale=sigma_z, size=(n_simulations, n_points, 1))
        points = np.append(points, z_coordinate, axis=2)
        volume = np.array([spatial.ConvexHull(pts).volume for pts in points])
        volume = volume.reshape(1, n_simulations)
        array_sim_volume = np.append(array_sim_volume, volume, 0)

    return array_sim_volume


def calc_volume_convex_hull_3d_pt1(n):
    """
    Calculate the expected value of the volume of the convex hull of n normal distributed points in 3D.
    The standard deviation is 1.

    Parameters
    ----------
    n : int
        Amount of normal distributed points.

    Returns
    -------
    float
    """
    return 8/3 * ((np.pi**1.5)/special.gamma(1.5)) * special.binom(n, 4) * integrate.quad(
        lambda x: cdf_normal(x) ** (n - 4) * pdf_normal(x) ** 4, -np.inf, np.inf)[0]


def calc_volume_convex_hull_3d_pt2(sigma_xy, sigma_z, n):
    """
    Calculate the expected value of the volume of the convex hull of n normal distributed points in 3D.

    Parameters
    ----------
    sigma_xy : float
        Standard deviation of the normal distribution in x,y plane.
    sigma_z : float
        Standard deviation of the normal distribution in z direction.
    n : int, array_like
        Amount of normal distributed points.

    Returns
    -------
    float or array
    """
    calc_volume_convex_hull_3d_pt1_array = np.vectorize(calc_volume_convex_hull_3d_pt1)

    return sigma_xy**2 * sigma_z * calc_volume_convex_hull_3d_pt1_array(n)


def sided_deviations(array_sim_values):
    """
    Calculate the mean and the standard deviation above and below the mean.

    Parameters
    ----------
    array_sim_values : numpy.ndarray
        Array of shape ((n_max_points - n_points_start), n_simulations).

    Returns
    -------
    std_high : numpy.ndarray
        Contains the standard deviations of all n, only taking the values above the mean.
    std_low : numpy.ndarray
        Contains the standard deviations of all n, only taking the values below the mean.
    means : numpy.ndarray
        Contains the mean of all n.
    """
    # noinspection PyTypeChecker
    means = np.mean(array_sim_values, axis=1)
    high_values = [array_sim_values[i][array_sim_values[i] > mean] for i, mean in enumerate(means)]
    low_values = [array_sim_values[i][array_sim_values[i] < mean] for i, mean in enumerate(means)]

    dev_high = [(high_values[i] - mean)**2 for i, mean in enumerate(means)]
    dev_low = [(low_values[i] - mean)**2 for i, mean in enumerate(means)]

    var_high = [dev_high[i].mean() for i, _ in enumerate(means)]
    var_low = [dev_low[i].mean() for i, _ in enumerate(means)]

    std_high = np.sqrt(np.array(var_high))
    std_low = np.sqrt(np.array(var_low))

    return std_high, std_low, means


def quantile(array_sim_values, quant):
    """
    Calculate the quant quantile of the simulated values.

    Parameters
    ----------
    array_sim_values : numpy.ndarray
        Array of shape ((n_max_points - n_points_start), n_simulations).
    quant : float
        Quantile to calculate.

    Returns
    -------
    quant_array : numpy.ndarray
        Array of shape ((n_max_points - n_points_start), ).
    """
    array_sim_values = np.sort(array_sim_values)
    quant = 1-quant
    quant_array = np.min(array_sim_values[:, -int(array_sim_values.shape[1]*quant):], axis=1)

    return quant_array


def lookuptable(measure, seed, n_max_points, n_simulations, sigma, quantile_1, quantile_2, save_to=None):
    """
    Receive lookup table of different statistical quantities of the measures of the convex hull.
    High n_simulations ensure a good approximation of the truth.

    Parameters
    ----------
    measure : str
        One of "peri_2d", "area_2d", "area_3d", "vol_3d".
    seed : int, array_like, numpy.random.BitGenerator
        Seed of random number generation.
    n_max_points : int
        Maximum amount of points.
    n_simulations : int
        Amount of simulations.
    sigma : float
        Standard deviation of the normal distribution in x,y (and z) plane.
    quantile_1 : float
        Number between 0 and 1. First quantile to be computed.
    quantile_2 : float
        Number between 0 and 1. Second quantile to be computed.
    save_to : path, default=None
        If path is provided, the dataframe is saved in the specified format.

    Returns
    -------
    lookup_table : pandas.core.frame.DataFrame
        Contains the calculated mean, simulated mean, positive and negative standard deviation as well as two variable
        quantiles.
    """
    rng = np.random.default_rng(seed)
    if measure == "peri_2d":
        index_1 = "$E(P_{ch}(n))$"
        index_2 = "$Mean(P_{ch}(n))$"
        n = np.arange(3, n_max_points+1, 1)
        array_sim_values = sim_peri_convex_hull_2d(n_max_points, rng, n_simulations, sigma)
        array_calc_values = calc_peri_convex_hull_2d_pt2(sigma, n)
    elif measure == "area_2d":
        index_1 = "$E(A_{ch}(n))$"
        index_2 = "$Mean(A_{ch}(n))$"
        n = np.arange(3, n_max_points+1, 1)
        array_sim_values = sim_area_convex_hull_2d(n_max_points, rng, n_simulations, sigma)
        array_calc_values = calc_area_convex_hull_2d_pt2(sigma, n)
    elif measure == "area_3d":
        index_1 = "$E(SA_{ch}(n))$"
        index_2 = "$Mean(SA_{ch}(n))$"
        n = np.arange(4, n_max_points+1, 1)
        array_sim_values = sim_area_convex_hull_3d(n_max_points, rng, n_simulations, sigma, sigma)
        array_calc_values = calc_area_convex_hull_3d_pt2(sigma, sigma, n)
    elif measure == "vol_3d":
        index_1 = "$E(V_{ch}(n))$"
        index_2 = "$Mean(V_{ch}(n))$"
        n = np.arange(4, n_max_points+1, 1)
        array_sim_values = sim_volume_convex_hull_3d(n_max_points, rng, n_simulations, sigma, sigma)
        array_calc_values = calc_volume_convex_hull_3d_pt2(sigma, sigma, n)
    else:
        raise("measure has to be one of " + '"peri_2d", ' + '"area_2d",' + '"area_3d",' + '"vol_3d".')

    std_pos, std_neg, mean = sided_deviations(array_sim_values)

    quant_1 = quantile(array_sim_values, quantile_1)
    quant_2 = quantile(array_sim_values, quantile_2)

    data = [array_calc_values, mean, std_pos, std_neg, quant_1, quant_2]
    index = [index_1, index_2, r"$std_{pos}(n)$", r"$std_{neg}(n)$",
             f"$quant_{{{quantile_1}}}(n)$", f"$quant_{{{quantile_2}}}(n)$"]
    columns = n
    lookup_table = pd.DataFrame(data=data, index=index, columns=columns)
    lookup_table.columns.names = ["$n$"]
    lookup_table = lookup_table.applymap("{:.2f}".format)
    if save_to is not None:
        lookup_table.to_csv(save_to)

    return lookup_table


# Since the lookup tables can be very time demanding due to standard error of mean reduction by high n_simulations,
# here an alternative approach is given exploiting ray to make use of all logical cores.


def generate_points_2d(n_max_points, seed, n_simulations, sigma):
    rng = np.random.default_rng(seed)
    points = []
    n_points = 2
    for i in range(n_max_points - 2):
        n_points += 1
        pts = rng.normal(loc=0, scale=sigma, size=(n_simulations, n_points, 2))
        points.append(pts)

    return points


def generate_points_3d(n_max_points, seed, n_simulations, sigma_xy, sigma_z):
    rng = np.random.default_rng(seed)
    points = []
    n_points = 3
    for i in range(n_max_points - 3):
        n_points += 1
        pts = rng.normal(loc=0, scale=sigma_xy, size=(n_simulations, n_points, 2))
        z_coordinate = rng.normal(loc=0, scale=sigma_z, size=(n_simulations, n_points, 1))
        pts = np.append(pts, z_coordinate, axis=2)
        points.append(pts)

    return points


@ray.remote
def perimeter_ray(points):
    perimeter = np.array([spatial.ConvexHull(pts).area for pts in points])

    return perimeter


@ray.remote
def area_ray(points):
    area = np.array([spatial.ConvexHull(pts).volume for pts in points])

    return area


@ray.remote
def surface_area_ray(points):
    surf_area = np.array([spatial.ConvexHull(pts).area for pts in points])

    return surf_area


@ray.remote
def volume_ray(points):
    vol = np.array([spatial.ConvexHull(pts).volume for pts in points])

    return vol


def lookuptable_ray(measure, seed, n_max_points, n_simulations_pt_1, n_simulations_pt_2, sigma, quantile_1, quantile_2,
                    save_to):
    rng = np.random.default_rng(seed)
    if measure == "peri_2d":
        index_1 = "$E(P_{ch}(n))$"
        index_2 = "$Mean(P_{ch}(n))$"
        array_sim_values = np.empty((n_max_points-2, 0))
        n = np.arange(3, n_max_points + 1, 1)
        for i in range(n_simulations_pt_1):
            points = generate_points_2d(n_max_points, rng, n_simulations_pt_2, sigma)
            future = [perimeter_ray.remote(point) for point in points]
            array = np.array(ray.get(future))
            array_sim_values = np.concatenate((array_sim_values, array), axis=1)
        array_calc_values = calc_peri_convex_hull_2d_pt2(sigma, n)
    elif measure == "area_2d":
        index_1 = "$E(A_{ch}(n))$"
        index_2 = "$Mean(A_{ch}(n))$"
        array_sim_values = np.empty((n_max_points-2, 0))
        n = np.arange(3, n_max_points + 1, 1)
        for i in range(n_simulations_pt_1):
            points = generate_points_2d(n_max_points, rng, n_simulations_pt_2, sigma)
            future = [area_ray.remote(point) for point in points]
            array = np.array(ray.get(future))
            array_sim_values = np.concatenate((array_sim_values, array), axis=1)
        array_calc_values = calc_area_convex_hull_2d_pt2(sigma, n)
    elif measure == "area_3d":
        index_1 = "$E(SA_{ch}(n))$"
        index_2 = "$Mean(SA_{ch}(n))$"
        array_sim_values = np.empty((n_max_points-3, 0))
        n = np.arange(4, n_max_points + 1, 1)
        for i in range(n_simulations_pt_1):
            points = generate_points_3d(n_max_points, rng, n_simulations_pt_2, sigma, sigma)
            future = [surface_area_ray.remote(point) for point in points]
            array = np.array(ray.get(future))
            array_sim_values = np.concatenate((array_sim_values, array), axis=1)
        array_calc_values = calc_area_convex_hull_3d_pt2(sigma, sigma, n)
    elif measure == "vol_3d":
        index_1 = "$E(V_{ch}(n))$"
        index_2 = "$Mean(V_{ch}(n))$"
        array_sim_values = np.empty((n_max_points-3, 0))
        n = np.arange(4, n_max_points + 1, 1)
        for i in range(n_simulations_pt_1):
            points = generate_points_3d(n_max_points, rng, n_simulations_pt_2, sigma, sigma)
            future = [volume_ray.remote(point) for point in points]
            array = np.array(ray.get(future))
            array_sim_values = np.concatenate((array_sim_values, array), axis=1)
        array_calc_values = calc_volume_convex_hull_3d_pt2(sigma, sigma, n)
    else:
        raise ("measure has to be one of " + '"peri_2d", ' + '"area_2d",' + '"area_3d",' + '"vol_3d".')

    std_pos, std_neg, mean = sided_deviations(array_sim_values)

    quant_1 = quantile(array_sim_values, quantile_1)
    quant_2 = quantile(array_sim_values, quantile_2)

    data = [array_calc_values, mean, std_pos, std_neg, quant_1, quant_2]
    index = [index_1, index_2, r"$std_{pos}(n)$", r"$std_{neg}(n)$",
             f"$quant_{{{quantile_1}}}(n)$", f"$quant_{{{quantile_2}}}(n)$"]
    columns = n
    lookup_table = pd.DataFrame(data=data, index=index, columns=columns)
    lookup_table.columns.names = ["$n$"]
    lookup_table = lookup_table.applymap("{:.2f}".format)
    if save_to is not None:
        lookup_table.to_csv(save_to)

    return lookup_table
