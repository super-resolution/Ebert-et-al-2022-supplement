import numpy as np
import lmfit as lm
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
import py_files.convex_hull as ch
import py_files.simulations as si
import py_files.cluster_properties as cp


def std_determination_lmfit(sample_count, area_chs, minimum, maximum, lookup_table_path):
    """
    Receive a prediction of the standard deviation of the normal distributed samples based on the distribution of the
    areas of the convex hull of the investigated clusters.

    Parameters
    ----------
    sample_count : numpy.ndarray
        Contains the sample count of each unique label of the algorithm.
    area_chs : numpy.ndarray
        Contains the area of the convex hull of each unique label of the algorithm.
    minimum : int
        Minimum value of sample_count for the cluster to be included in the determination of the standard deviation.
    maximum : int
        Maximum value of sample_count for the cluster to be included in the determination of the standard deviation.
    lookup_table_path : str
        The directory of the lookup_table of areas of the convex hull.

    Returns
    -------
    pred_sigma : float
        Value of the determined predicted standard deviation.
    num_clust : int
        Total amount of clusters used for the determination.
    """
    lookup_table = pd.read_csv(lookup_table_path, index_col=0)
    std_pos = lookup_table.loc["$std_{pos}(n)$"].astype(float).values
    std_neg = lookup_table.loc["$std_{neg}(n)$"].astype(float).values
    std = np.mean(np.array([std_pos, std_neg]), axis=0)
    weights = 1/std[minimum-3:maximum-3+1]
    n = np.arange(minimum, maximum + 1)
    means = []
    num_clust = 0
    drop_indices = []
    for i, n_ in enumerate(n):
        indices = np.where(sample_count == n_)[0]
        areas = area_chs[indices]
        num_clust += len(areas)
        if len(areas) > 0:
            mean_area = np.mean(areas)
            means.append(mean_area)
        else:
            drop_indices.append(i)

    n = np.delete(n, drop_indices, axis=0)
    weights = np.delete(weights, drop_indices, axis=0)
    model = lm.Model(ch.calc_area_convex_hull_2d_pt2, independent_vars=["n"])
    params = lm.Parameters()
    params.add("sigma", value=5, min=1, max=20)
    result = model.fit(means, n=n, params=params, weights=weights)
    pred_sigma = result.params["sigma"].value

    return pred_sigma, num_clust


def std_determination_per_n(n_simulations, parent_intensities, region_limits, cluster_std, sim_param, alg_param, n,
                            lookup_table_path):
    """
    Receive two arrays representing the mean and standard deviation of the prediction via std_determination_lmfit
    relative to the true cluster_std. A value for each combination of entries of parent_intensities and n will be
    generated.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    parent_intensities : list
        Contains all parent intensity values to be analyzed.
    region_limits : list
        Contains region limit values for each of parent_intensities, e.g. the result of simulations.limits.
    cluster_std : float
        The standard deviation of the normal distribution with the parent points as mean.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_dstorm except parent_intensity, lower_limit, upper_limit
        and cluster_std.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN.
    n : list
        Contains each value of sample_count to be analyzed.
    lookup_table_path : str
        The directory of the lookup_table of areas of the convex hull.

    Returns
    -------
    means : numpy.ndarray
        Contains the means of pred_sigma/cluster_std. pred_sigma originates from std_determination_lmfit.
        Shape (parent_intensities, n).
    stds : numpy.ndarray
        Contains the stds of pred_sigma/cluster_std. pred_sigma originates from std_determination_lmfit.
        Shape (parent_intensities, n).
    """
    array = np.empty((n_simulations, len(parent_intensities), len(n)))
    for i in range(n_simulations):
        for j, (parent_intensity, limit) in enumerate(zip(parent_intensities, region_limits)):
            samples, labels, _ = si.sim_dstorm(parent_intensity=parent_intensity, lower_limit=-limit,
                                               upper_limit=limit, cluster_std=cluster_std, **sim_param)
            clust_labels = DBSCAN(**alg_param).fit_predict(samples)
            _, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
            for h, n_ in enumerate(n):
                pred_sigma, num_clust = std_determination_lmfit(sample_count, area_chs, minimum=n_, maximum=n_,
                                                                lookup_table_path=lookup_table_path)
                array[i, j, h] = pred_sigma/cluster_std

    means = np.nanmean(array, axis=0)
    stds = np.nanstd(array, axis=0)

    return means, stds


def std_determination_per_cluster_count(n_simulations, cluster_std, sim_param, alg_param, samplings, n, seed,
                                        lookup_table_path):
    """
    Receive two arrays representing the mean and standard deviation of the prediction via std_determination_lmfit
    relative to the true cluster_std. Each entry represents the result using only a limited amount of clusters (see
    samplings).
    Note: the limits and parent_intensity combination of sim_param has to be chosen such that at least np.max(samplings)
    of clusters with sample_count equal to n are present.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    cluster_std : float
        The standard deviation of the normal distribution with the parent points as mean.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_dstorm except cluster_std.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN
    samplings : list
        Contains each value of cluster counts to be analyzed.
    n : numpy.ndarray
        Contains each value of sample_count to be used for analysis.
    seed : numpy.random.BitGenerator, int or numpy.ndarray
        Seed of random number generation.
    lookup_table_path : str
        The directory of the lookup_table of areas of the convex hull.

    Returns
    -------
    means : numpy.ndarray
        Contains the means of pred_sigma/cluster_std. pred_sigma originates from std_determination_lmfit.
        Shape (samplings,).
    stds : numpy.ndarray
        Contains the stds of pred_sigma/cluster_std. pred_sigma originates from std_determination_lmfit.
        Shape (samplings,).
    """
    rng = np.random.default_rng(seed)
    array = np.empty((n_simulations, len(samplings)))
    for i in range(n_simulations):
        samples, labels, _ = si.sim_dstorm(cluster_std=cluster_std, **sim_param)
        clust_labels = DBSCAN(**alg_param).fit_predict(samples)
        _, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
        indices = np.where(np.in1d(sample_count, n))[0]
        for j, sampling in enumerate(samplings):
            indices_ = rng.choice(indices, sampling, replace=False)
            areas = area_chs[indices_]
            sample_counts = sample_count[indices_]
            pred_sigma, num_clust = std_determination_lmfit(sample_counts, areas, minimum=np.min(n),
                                                            maximum=np.max(n), lookup_table_path=lookup_table_path)
            array[i, j] = pred_sigma/cluster_std

    means = np.nanmean(array, axis=0)
    stds = np.nanstd(array, axis=0)

    return means, stds


def std_determination_per_epsilon(n_simulations, cluster_std, sim_param, alg_param, epsilons, lookup_table_path,
                                  minimum, maximum):
    """
    Receive six arrays representing the mean and the standard deviation of the following measures:
    std_score  - the ratio of the prediction via std_determination_lmfit to the true cluster_std;
    ari - the adjusted rand index of the clustering;
    correct_clust - the result of cluster_properties.correct_clusters;

    The entries of each array represent the analyzed clustering result using the entries of epsilons.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    cluster_std : float
        The standard deviation of the normal distribution with the parent points as mean.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_dstorm except cluster_std.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN except eps.
    epsilons : list
        Contains each value of DBSCANs eps to be analyzed.
    lookup_table_path : str
        The directory of the lookup_table of areas of the convex hull.
    minimum : int
        Minimum value of sample_count for the cluster to be included in the determination of the standard deviation.
    maximum : int
        Maximum value of sample_count for the cluster to be included in the determination of the standard deviation.

    Returns
    -------
    std_score_mean : numpy.ndarray
        Contains the means of pred_sigma/cluster_std. pred_sigma originates from std_determination_lmfit.
        Shape (epsilons,)
    std_score_std : numpy.ndarray
        Contains the stds of pred_sigma/cluster_std. pred_sigma originates from std_determination_lmfit.
        Shape (epsilons,)
    ari_mean : numpy.ndarray
        Contains the means of sklearn.metrics.cluster.adjusted_rand_score.
        Shape (epsilons,)
    ari_std : numpy.ndarray
        Contains the stds of sklearn.metrics.cluster.adjusted_rand_score.
        Shape (epsilons,)
    correct_clust_mean : numpy.ndarray
        Contains the means of cluster_property.correct_clusters.
        Shape (epsilons,)
    correct_clust_std : numpy.ndarray
        Contains the stds of cluster_property.correct_clusters.
        Shape (epsilons,)
    """
    std_scores = np.empty((n_simulations, len(epsilons)))
    aris = np.empty((n_simulations, len(epsilons)))
    correct_clust = np.empty((n_simulations, len(epsilons)))
    for i in range(n_simulations):
        samples, labels, _ = si.sim_dstorm(cluster_std=cluster_std, **sim_param)
        for j, eps in enumerate(epsilons):
            clust_labels = DBSCAN(eps=eps, **alg_param).fit_predict(samples)
            _, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
            pred_sigma, num_clust = std_determination_lmfit(sample_count, area_chs, minimum=minimum, maximum=maximum,
                                                            lookup_table_path=lookup_table_path)
            ari = adjusted_rand_score(labels, clust_labels)
            wrong_cluster_count, _ = cp.wrong_clusters(clust_labels, labels)
            missed_cluster_count, _ = cp.missed_clusters(clust_labels, labels)
            subdiv_cluster_count, _ = cp.subdivided_clusters(clust_labels, labels)
            merg_cluster_count, _ = cp.merged_clusters(clust_labels, labels)
            correct_cluster = cp.correct_clusters(clust_labels, wrong_cluster_count, missed_cluster_count,
                                                  subdiv_cluster_count, merg_cluster_count)

            std_scores[i, j] = pred_sigma/cluster_std
            aris[i, j] = ari
            correct_clust[i, j] = correct_cluster

    std_score_mean = np.nanmean(std_scores, axis=0)
    std_score_std = np.nanstd(std_scores, axis=0)
    ari_mean = np.nanmean(aris, axis=0)
    ari_std = np.nanstd(aris, axis=0)
    correct_clust_mean = np.nanmean(correct_clust, axis=0)
    correct_clust_std = np.nanstd(correct_clust, axis=0)

    return std_score_mean, std_score_std, ari_mean, ari_std, correct_clust_mean, correct_clust_std
