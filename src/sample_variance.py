import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import src.simulations as si
import src.convex_hull as ch
import src.cluster_properties as cp


def simulate_variance_estimation(n_max_points, n_simulations, sigma, seed, bias=True, different_x_y=True):
    """
    Simulate n normal distributed points in 2D and calculate their variance.

    Parameters
    ----------
    n_max_points : int
        Maximum amount of points.
    n_simulations : int
        Amount of simulations.
    sigma : float
        Standard deviation of the normal distribution.
    seed : int, array_like, numpy.random.BitGenerator
        Seed of random number generation.
    bias : bool
        Whether to determine the biased or the unbiased sample variance.
    different_x_y : bool
        Whether to simulate normal distributed points around the centroid (x=a, y=a) or (x=a, y=b).

    Returns
    -------
    array_sim_variance : numpy.ndarray
        Array of shape ((n_max_points - n_points_start), n_simulations).
    """
    rng = np.random.default_rng(seed)
    array_sim_variance = np.empty((0, n_simulations))
    n_points = 2
    if bias:
        ddof = 0
    else:
        ddof = 1
    for i in range(n_max_points-2):
        n_points += 1
        if different_x_y:
            points = rng.normal(loc=(4, 8), scale=sigma, size=(n_simulations, n_points, 2))
            var = np.array([np.mean(np.var(pts, ddof=ddof, axis=0)) for pts in points])
        else:
            points = rng.normal(loc=0, scale=sigma, size=(n_simulations, n_points, 2))
            var = np.array([np.var(pts, ddof=ddof) for pts in points])
        var = var.reshape(1, n_simulations)
        array_sim_variance = np.append(array_sim_variance, var, 0)

    return array_sim_variance


def observed_mse(array_sim_variance, sigma, bias=True):
    """
    Determines the mean squared error MSE of variance estimation.

    Parameters
    ----------
    array_sim_variance : numpy.ndarray
        Array of shape ((n_max_points - n_points_start), n_simulations). Contains the variance estimations.
    sigma : float
        Standard deviation of the normal distribution whose variance was estimated.
    bias : bool
        Whether to determine the MSE of a biased or unbiased sample variance estimator.

    Returns
    -------
    mse_obs : numpy.ndarray
        Contains the MSE of the variance estimation.
    """
    var_obs = np.var(array_sim_variance, axis=1)
    if bias:
        means = np.mean(array_sim_variance, axis=1)
        bias = means - sigma**2
        mse_obs = var_obs + bias**2
    else:
        mse_obs = var_obs

    return mse_obs


def expected_value_variance(n, sigma, bias=True, different_x_y=True):
    """
    Calculates the expected value of the variance estimation of normal distributed 2D points.

    Parameters
    ----------
    n : int, collection
        Number of points (coordinate-pairs) to estimate the variance.
    sigma : float
        The standard deviation of the normal distributed points.
    bias : bool
        Whether to use the biased or unbiased variance estimation.
    different_x_y : bool
        Whether the normal distributed x and y coordinates share a common value as origin (e.g., (0, 0), or (3.4, 3.4)).

    Returns
    -------
    expected_value : float, collection
        The expected value of the variance estimation.
    """
    if bias:
        if different_x_y:
            expected_value = sigma**2 * (1 - 1/n)
        else:
            n = n*2
            expected_value = sigma**2 * (1 - 1/n)
    else:  # note that here, it doesn't matter if x and y are different because the expected value will always be
        # sigma**2
        expected_value = sigma**2 * (1 - 1/n) * n/(n-1)

    return expected_value


def mse_calculation(n, sigma, bias=True, different_x_y=True):
    """
    Calculates the mean squared error of the variance estimation of normal distributed 2D points.

    Parameters
    ----------
    n : int, collection
        Number of points (coordinate-pairs) to estimate the variance.
    sigma : float
        The standard deviation of the normal distributed points.
    bias : bool
        Whether to use the biased or unbiased variance estimation.
    different_x_y : bool
        Whether the normal distributed x and y coordinates share a common value as origin (e.g., (0, 0) or (3.4, 3.4)).

    Returns
    -------
    mse : float, collection
        The mean squared error of the variance estimation.
    """
    if bias:
        if different_x_y:
            expected_value = expected_value_variance(n, sigma, bias=True)
            bias_sq = (expected_value - sigma**2)**2
            mse = ((2*n - 1) / n**2 * sigma**4 - bias_sq) / 2 + bias_sq  # division by 2 because each variance value is
            # the result of the mean of 2 variance values -- but the bias part of the mse has to subtracted beforehand,
            # because this is not influences by the averaging -- then add it again
        else:
            n = n*2  # every coodinate consists of 2 values, e.g., 3 points lead to 6 measures used to estimate the
            # estimatot (here the variance)
            mse = (2*n - 1) / n**2 * sigma**4
    else:
        if different_x_y:
            mse = (2 / (n-1) * sigma**4) / 2  # division by 2 because each variance value is the result of the mean
            # of 2 variance values
        else:
            n = n*2  # every coodinate consists of 2 values, e.g., 3 points lead to 6 measures used to estimate the
            # estimatot (here the variance)
            mse = 2 / (n-1) * sigma**4

    return mse


def lookuptable_variance(bias, seed, n_max_points, n_simulations, sigma, quantiles, save_to):
    """
    Receive lookup table of different statistical quantities of the sample variance estimation.
    High n_simulations ensure a good approximation of the truth.

    Parameters
    ----------
    bias : bool
        Whether to use the biased or unbiased sample variance estimation.
    seed : int, array_like, numpy.random.BitGenerator
        Seed of random number generation.
    n_max_points : int
        Maximum amount of points.
    n_simulations : int
        Amount of simulations.
    sigma : float
        Standard deviation of the normal distribution.
    quantiles : collection
        Contains the quantile values to be included as statistical quantity.
    save_to : path
        If path is provided, the dataframe is saved in the specified format.

    Returns
    -------
    lookuptable : pandas.core.frame.DataFrame
        Contains the calculated mean, simulated mean, positive and negative standard deviation as well as the specified
        quantiles.
    """
    rng = np.random.default_rng(seed)
    index_1 = "$E(var(n))$"
    index_2 = "$Mean(var(n))$"
    index_3 = "$std_{pos}(n)$"
    index_4 = "$std_{neg}(n)$"
    n = np.arange(3, n_max_points + 1, 1)
    array_sim_values = np.empty((0, n_simulations))

    n_points = 2
    for i in range(n_max_points - 2):
        n_points += 1
        ptss = rng.normal(loc=(4, 8), scale=sigma, size=(n_simulations, n_points, 2))

        if bias:
            var = np.array([np.mean(np.var(pts, ddof=0, axis=0)) for pts in ptss])
        else:
            var = np.array([np.mean(np.var(pts, ddof=1, axis=0)) for pts in ptss])

        var = var.reshape(1, n_simulations)
        array_sim_values = np.append(array_sim_values, var, axis=0)

    std_pos, std_neg, mean = ch.sided_deviations(array_sim_values)
    array_calc_values = expected_value_variance(n, sigma, bias, different_x_y=True)

    data = [array_calc_values, mean, std_pos, std_neg]
    index = [index_1, index_2, index_3, index_4]

    for quantile in quantiles:
        quant = ch.quantile(array_sim_values, quantile)
        data.append(quant)
        index.append(f"$quant_{{{quantile}}}(n)$")

    columns = n
    lookuptable = pd.DataFrame(data=data, index=index, columns=columns)
    lookuptable.columns.names = ["$n$"]
    lookuptable = lookuptable.applymap("{:.2f}".format)
    if save_to is not None:
        lookuptable.to_csv(save_to)

    return lookuptable


def lookuptable_convexhull(mode, seed, n_max_points, n_simulations, sigma, quantiles, save_to):
    """
    Receive lookup table of different statistical quantities of the measures of the convex hull.
    High n_simulations ensure a good approximation of the truth.

    Parameters
    ----------
    mode : str
        Either "area" or "peri".
    seed : int, array_like, numpy.random.BitGenerator
        Seed of random number generation.
    n_max_points : int
        Maximum amount of points.
    n_simulations : int
        Amount of simulations.
    sigma : float
        Standard deviation of the normal distribution.
    quantiles : collection
        Contains the quantile values to be included as statistical quantity.
    save_to : path
        If path is provided, the dataframe is saved in the specified format.

    Returns
    -------
    lookuptable : pandas.core.frame.DataFrame
        Contains the calculated mean, simulated mean, positive and negative standard deviation as well as the specified
        quantiles.
    """
    rng = np.random.default_rng(seed)
    index_3 = "$std_{pos}(n)$"
    index_4 = "$std_{neg}(n)$"
    n = np.arange(3, n_max_points + 1, 1)

    if mode == "area":
        index_1 = "$E(A(n))$"
        index_2 = "$Mean(A(n))$"
        array_sim_values = ch.sim_area_convex_hull_2d(n_max_points, rng, n_simulations, sigma)
        array_calc_values = ch.calc_area_convex_hull_2d_pt2(sigma, n)
    elif mode == "peri":
        index_1 = "$E(P(n))$"
        index_2 = "$Mean(P(n))$"
        array_sim_values = ch.sim_peri_convex_hull_2d(n_max_points, rng, n_simulations, sigma)
        array_calc_values = ch.calc_peri_convex_hull_2d_pt2(sigma, n)
    else:
        raise ValueError

    std_pos, std_neg, mean = ch.sided_deviations(array_sim_values)

    data = [array_calc_values, mean, std_pos, std_neg]
    index = [index_1, index_2, index_3, index_4]

    for quantile in quantiles:
        quant = ch.quantile(array_sim_values, quantile)
        data.append(quant)
        index.append(f"$quant_{{{quantile}}}(n)$")

    columns = n
    lookuptable = pd.DataFrame(data=data, index=index, columns=columns)
    lookuptable.columns.names = ["$n$"]
    lookuptable = lookuptable.applymap("{:.2f}".format)
    if save_to is not None:
        lookuptable.to_csv(save_to)

    return lookuptable


def large_cluster_variance(unique_labels, sample_count, variances, max_variances):
    """
    Receive the clusters with sample variances larger than some threshold values (max_variances).

    Parameters
    ----------
    unique_labels : numpy.ndarray
        First result of cluster_properties.cluster_property.
    sample_count : numpy.ndarray
        Second result of cluster_properties.cluster_property.
    variances : numpy.ndarray
        Result of get_variances.
    max_variances : numpy.ndarray
        Threshold variances. Can be generated using lookup_tables.

    Returns
    -------
    larger_cluster_score : float
        The amount of large clusters relative to examined clusters.
    large_cluster : numpy.ndarray
        Labels of large clusters.
    """
    cluster_count = 0
    large_cluster_count = 0
    large_cluster = []

    for i, num in enumerate(max_variances):
        indices = np.where(sample_count == i + 3)[0]
        variance = variances[indices]
        large = np.where(variance > num)[0]
        idx = indices[large]
        labels = unique_labels[idx]
        large_cluster.append(labels)
        large_cluster_count += len(labels)
        cluster_count += len(variance)

    idx = np.where(sample_count > 200)[0]
    labels = unique_labels[idx]
    large_cluster.append(labels)
    large_cluster_count += len(labels)
    cluster_count += len(labels)

    if cluster_count != 0:
        large_cluster_score = large_cluster_count / cluster_count
    else:
        large_cluster_score = 0
    large_cluster = np.concatenate(large_cluster)

    return large_cluster_score, large_cluster


def get_variance(clust_labels, samples, bessel_correction=True):
    """
    Receive the sample variances of clusters.

    Parameters
    ----------
    clust_labels : numpy.ndarray
        Labels resulting from clustering.
    samples : numpy.ndarray
        Clustered samples.
    bessel_correction : bool
        Whether the variance estimation will be biased or bessel-corrected (unbiased).

    Returns
    -------
    variances : numpy.ndarray
        Contains the sample variance of each of unique_labels.
    """
    variances = []

    unique_labels = np.unique(clust_labels)
    unique_labels = np.delete(unique_labels, np.where(unique_labels == -1))
    for i in unique_labels:
        indices_cluster = np.where(clust_labels == i)
        samples_cluster = samples[indices_cluster]
        if bessel_correction:
            variance = np.mean(np.var(samples_cluster, ddof=1, axis=0))
            # note: if done manually (not with numpy) and the mean
            # coordinate is estimated along 0th axis, one has to take the sum along axis 0 as well:
            # mean_coordinate = np.mean(samples_cluster, axis=0)
            # delta = samples_cluster - mean_coordinate
            # delta_sq = delta**2
            # sum_ = np.sum(delta_sq, axis=0)
            # variance = sum_ / (len(samples_cluster) / 2 - 1)  # /2 due to only x or y values are taken
        else:
            variance = np.mean(np.var(samples_cluster, axis=0))  # note that in this case, the variance is not equal
            # the to mean squared error, because the bias has to be taken into account

        variances.append(variance)

    variances = np.array(variances)

    return variances


def selection_by_variance(n_simulations, parent_intensities, region_limits, sim_param, alg_param, threshold_var,
                          bessel_correction):
    """
    Receive two arrays representing the mean and std of the following measures:
    FDR - False discovery rate: The amount of false positives relative to the positives (FP & TP, i.e. large clusters);
    TPR - True positive rate: The amount of true positives relative to the condition positive (i.e. merged clusters);

    Large clusters are determined via the sample variance.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    parent_intensities : list
        Contains all parent intensity values to be analyzed.
    region_limits : list
        Contains region limit values for each of parent_intensities, e.g. the result of simulations.limits.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_dstorm except parent_intensity, lower_limit, upper_limit.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN.
    threshold_var : list
        Contains threshold values for sample variances.
    bessel_correction : bool
        Whether the variance estimation will be biased or bessel-corrected (unbiased).

    Returns
    -------
    means : numpy.ndarray
        Contains the means of FDR and TPR.
        Shape (parent_intensities, max_sample_counts, (FDR, TPR)).
    stds : numpy.ndarray
        Contains the stds of FDR and TPR.
        Shape (parent_intensities, max_sample_counts, (FDR, TPR)).
    """
    array = np.empty((n_simulations, len(parent_intensities), len(threshold_var), 2))
    for i in range(n_simulations):
        for j, (parent_intensity, limit) in enumerate(zip(parent_intensities, region_limits)):
            samples, labels, _ = si.sim_dstorm(parent_intensity=parent_intensity, lower_limit=-limit, upper_limit=limit,
                                               **sim_param)
            clust_labels = DBSCAN(**alg_param).fit_predict(samples)
            unique_labels, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
            merg_cluster_count, merg_cluster = cp.merged_clusters(clust_labels, labels)
            var = get_variance(clust_labels, samples, bessel_correction)

            for h, threshold in enumerate(threshold_var):
                _, large_cluster = large_cluster_variance(unique_labels, sample_count, var, threshold)
                true_positives = np.where(np.in1d(large_cluster, merg_cluster))[0]
                if len(large_cluster) != 0:
                    false_discovery_rate = 1 - len(true_positives)/len(large_cluster)
                else:
                    false_discovery_rate = np.nan
                true_positive_rate = len(true_positives) / len(merg_cluster)

                array[i, j, h, 0] = false_discovery_rate
                array[i, j, h, 1] = true_positive_rate
    means = np.nanmean(array, axis=0)
    stds = np.nanstd(array, axis=0)

    return means, stds


def selection_by_ch_area(n_simulations, parent_intensities, region_limits, sim_param, alg_param, max_areas):
    """
    Receive two arrays representing the mean and std of the following measures:
    FDR - False discovery rate: The amount of false positives relative to the positives (FP & TP, i.e. large clusters);
    TPR - True positive rate: The amount of true positives relative to the condition positive (i.e. merged clusters);

    Large clusters are determined via the area of the convex hull.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    parent_intensities : list
        Contains all parent intensity values to be analyzed.
    region_limits : list
        Contains region limit values for each of parent_intensities, e.g. the result of simulations.limits.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_dstorm except parent_intensity, lower_limit, upper_limit.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN.
    max_areas: list
        Contains threshold values for the area of the convex hull.

    Returns
    -------
    means : numpy.ndarray
        Contains the means of FDR and TPR.
        Shape (parent_intensities, max_sample_counts, (FDR, TPR)).
    stds : numpy.ndarray
        Contains the stds of FDR and TPR.
        Shape (parent_intensities, max_sample_counts, (FDR, TPR)).
    """
    array = np.empty((n_simulations, len(parent_intensities), len(max_areas), 2))
    for i in range(n_simulations):
        for j, (parent_intensity, limit) in enumerate(zip(parent_intensities, region_limits)):
            samples, labels, _ = si.sim_dstorm(parent_intensity=parent_intensity, lower_limit=-limit, upper_limit=limit,
                                               **sim_param)
            clust_labels = DBSCAN(**alg_param).fit_predict(samples)
            unique_labels, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
            merg_cluster_count, merg_cluster = cp.merged_clusters(clust_labels, labels)

            for h, threshold in enumerate(max_areas):
                large_cluster_score, large_cluster = cp.large_clusters(unique_labels, sample_count, area_chs, threshold,
                                                                       n=None)
                true_positives = np.where(np.in1d(large_cluster, merg_cluster))[0]
                if large_cluster_score > 0:
                    false_discovery_rate = 1 - len(true_positives)/len(large_cluster)
                else:
                    false_discovery_rate = np.nan
                true_positive_rate = len(true_positives) / len(merg_cluster)

                array[i, j, h, 0] = false_discovery_rate
                array[i, j, h, 1] = true_positive_rate
    means = np.nanmean(array, axis=0)
    stds = np.nanstd(array, axis=0)

    return means, stds


# perimeter to compare with biased var, unbiased var and area

def cluster_property_peri(clust_labels, samples):
    """
    Receive several cluster properties: the unique labels, their associated sample counts, their associated perimeters
    of the convex hull as well as their center coordinates.

    Parameters
    ----------
    clust_labels : numpy.ndarray
        Labels resulting from clustering.
    samples : numpy.ndarray
        Clustered samples.

    Returns
    -------
    unique_labels : numpy.ndarray
        Contains all labels of clust_labels once except -1 (noise).
    sample_count : numpy.ndarray
        Contains the sample count of each of unique_labels.
    peri_chs : numpy.ndarray
        Contains the perimeter of the convex hull of each of unique_labels.
    coordinates : numpy.ndarray
        Contains the center coordinate of each of unique_labels.
    """
    sample_count = []
    peri_chs = []
    coordinates = []

    unique_labels = np.unique(clust_labels)
    unique_labels = np.delete(unique_labels, np.where(unique_labels == -1))
    for i in unique_labels:
        indices_cluster = np.where(clust_labels == i)
        samples_cluster = samples[indices_cluster]
        sample_count.append(len(samples_cluster))
        peri_ch = ConvexHull(samples_cluster).area
        peri_chs.append(peri_ch)
        coordinate = np.mean(samples_cluster, axis=0)
        coordinates.append(coordinate)

    sample_count = np.array(sample_count)
    peri_chs = np.array(peri_chs)
    coordinates = np.array(coordinates)

    return unique_labels, sample_count, peri_chs, coordinates


def large_clusters_peri(unique_labels, sample_count, peri_chs, max_peris):
    """
    Receive the clusters with perimeter of convex hull larger than some threshold values (max_peris).

    Parameters
    ----------
    unique_labels : numpy.ndarray
        First result of cluster_property_peri.
    sample_count : numpy.ndarray
        Second result of cluster_property_peri.
    peri_chs : numpy.ndarray
        Third result of cluster_property_peri.
    max_peris : numpy.ndarray
        Threshold perimeters. Can be generated using lookup_tables.

    Returns
    -------
    larger_cluster_score : float
        The amount of large clusters relative to examined clusters.
    large_cluster : numpy.ndarray
        Labels of large clusters.
    """
    cluster_count = 0
    large_cluster_count = 0
    large_cluster = []

    for i, num in enumerate(max_peris):
        indices = np.where(sample_count == i + 3)[0]
        peris = peri_chs[indices]
        large = np.where(peris > num)[0]
        idx = indices[large]
        labels = unique_labels[idx]
        large_cluster.append(labels)
        large_cluster_count += len(labels)
        cluster_count += len(peris)

    idx = np.where(sample_count > 200)[0]
    labels = unique_labels[idx]
    large_cluster.append(labels)
    large_cluster_count += len(labels)
    cluster_count += len(labels)
    if cluster_count != 0:
        large_cluster_score = large_cluster_count / cluster_count
    else:
        large_cluster_score = 0
    large_cluster = np.concatenate(large_cluster)

    return large_cluster_score, large_cluster


def selection_by_ch_peri(n_simulations, parent_intensities, region_limits, sim_param, alg_param, max_peris):
    """
    Receive two arrays representing the mean and std of the following measures:
    FDR - False discovery rate: The amount of false positives relative to the positives (FP & TP, i.e. large clusters);
    TPR - True positive rate: The amount of true positives relative to the condition positive (i.e. merged clusters);

    Large clusters are determined via the perimeter of the convex hull.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    parent_intensities : list
        Contains all parent intensity values to be analyzed.
    region_limits : list
        Contains region limit values for each of parent_intensities, e.g. the result of simulations.limits.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_dstorm except parent_intensity, lower_limit, upper_limit.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN.
    max_peris: list
        Contains threshold values for the perimeter of the convex hull.

    Returns
    -------
    means : numpy.ndarray
        Contains the means of FDR and TPR.
        Shape (parent_intensities, max_sample_counts, (FDR, TPR)).
    stds : numpy.ndarray
        Contains the stds of FDR and TPR.
        Shape (parent_intensities, max_sample_counts, (FDR, TPR)).
    """
    array = np.empty((n_simulations, len(parent_intensities), len(max_peris), 2))
    for i in range(n_simulations):
        for j, (parent_intensity, limit) in enumerate(zip(parent_intensities, region_limits)):
            samples, labels, _ = si.sim_dstorm(parent_intensity=parent_intensity, lower_limit=-limit, upper_limit=limit,
                                               **sim_param)
            clust_labels = DBSCAN(**alg_param).fit_predict(samples)
            unique_labels, sample_count, peri_chs, _ = cluster_property_peri(clust_labels, samples)
            merg_cluster_count, merg_cluster = cp.merged_clusters(clust_labels, labels)

            for h, threshold in enumerate(max_peris):
                large_cluster_score, large_cluster = large_clusters_peri(unique_labels, sample_count, peri_chs,
                                                                         threshold)
                true_positives = np.where(np.in1d(large_cluster, merg_cluster))[0]
                if large_cluster_score > 0:
                    false_discovery_rate = 1 - len(true_positives)/len(large_cluster)
                else:
                    false_discovery_rate = np.nan
                true_positive_rate = len(true_positives) / len(merg_cluster)

                array[i, j, h, 0] = false_discovery_rate
                array[i, j, h, 1] = true_positive_rate
    means = np.nanmean(array, axis=0)
    stds = np.nanstd(array, axis=0)

    return means, stds

