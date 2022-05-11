import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import py_files.simulations as si
import py_files.cluster_properties as cp


def selection_criterion_analysis(n_simulations, parent_intensities, region_limits, sim_param, alg_param, max_areas,
                                 max_areas_keys, n=None):
    """
    Receive three tables representing the mean, std and mean±std of the following measures:
    Ground truth merged clusters - The relative amount of clusters lost by the algorithm due to merging;
    Algorithm merged clusters - The relative amount of merged clusters of the algorithm;
    Correct clusters - see cluster_properties;
    TPR - True positive rate: The amount of true positives relative to the condition positive (i.e. merged clusters);
    FDR - False discovery rate: The amount of false positives relative to the positives (FP & TP, i.e. large clusters);
    Algorithm large clusters - The relative amount of identified large clusters of the algorithm.

    Large clusters are determined via the convex hull.

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
    max_areas : list
        Contains numpy.ndarrays representing threshold areas. Each entry can be generated using lookup_tables.
    max_areas_keys : list
        Contains names of threshold arrays.
    n : int, default=None
        If provided, n serves as a sample_count threshold, additionally to max_areas threshold.

    Returns
    -------
    means : pandas.core.frame.DataFrame
        Contains the means of each measure for each parent intensity (i.e. density) (level 0) and threshold (level 1)
        combination.
    stds : pandas.core.frame.DataFrame
        Contains the stds of each measure for each parent intensity (i.e. density)
        (level 0) and threshold (level 1)combination.
    representation : pandas.core.frame.DataFrame
        Combination of means and stds for visual purposes.
    """
    list_3 = []
    for i in range(n_simulations):
        list_2 = []
        for (parent_intensity, limit) in zip(parent_intensities, region_limits):
            samples, labels, _ = si.sim_dstorm(parent_intensity=parent_intensity, lower_limit=-limit, upper_limit=limit,
                                               **sim_param)
            unique_sim_labels = np.unique(np.delete(labels, np.where(labels == -1)))
            clust_labels = DBSCAN(**alg_param).fit_predict(samples)
            unique_labels, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)

            wrong_cluster_count, _ = cp.wrong_clusters(clust_labels, labels)
            missed_cluster_count, _ = cp.missed_clusters(clust_labels, labels)
            subdiv_cluster_count, _ = cp.subdivided_clusters(clust_labels, labels)
            merg_cluster_count, merg_cluster = cp.merged_clusters(clust_labels, labels)
            # merg_cluster represents the condition positive, i.e. the false negatives and true positives
            correct_clust_score = cp.correct_clusters(clust_labels, wrong_cluster_count, missed_cluster_count,
                                                      subdiv_cluster_count, merg_cluster_count)
            merg_cluster_sim_rel = merg_cluster_count / len(unique_sim_labels)
            merg_cluster_alg_rel = len(merg_cluster) / len(unique_labels)

            list_1 = []
            for max_area in max_areas:
                large_cluster_score, large_cluster = cp.large_clusters(unique_labels, sample_count, area_chs, max_area,
                                                                       n)
                # large_cluster represents the true positives (TP) and false positives (FP)
                large_cluster_rel = len(large_cluster) / len(unique_labels)  # equal to large_cluster_score, if n=None
                true_positives = np.where(np.in1d(large_cluster, merg_cluster))[0]
                if large_cluster_score > 0:
                    false_discovery_rate = 1 - len(true_positives)/len(large_cluster)  # i.e. 1 - PPV
                else:
                    false_discovery_rate = np.nan
                true_positive_rate = len(true_positives) / len(merg_cluster)

                columns = [merg_cluster_sim_rel, merg_cluster_alg_rel, correct_clust_score,
                           true_positive_rate, false_discovery_rate, large_cluster_rel]
                columnnames = ["Ground truth merged clusters", "Algorithm merged clusters",
                               "Correct clusters", "TPR", "FDR", "Algorithm large clusters"]
                table_1 = pd.DataFrame([columns], columns=columnnames)
                list_1.append(table_1)
            table_2 = pd.concat(list_1, keys=max_areas_keys, names=["Threshold"])
            list_2.append(table_2)
        table_3 = pd.concat(list_2, keys=parent_intensities, names=["Density"])
        list_3.append(table_3)
    table_4 = pd.concat(list_3, keys=np.arange(0, n_simulations+1))
    grouped_by_simulations = table_4.groupby(level=[1, 2])
    means = grouped_by_simulations.mean()
    stds = grouped_by_simulations.std()
    representation = pd.concat([means, stds]).applymap("{:.2f}".format)
    representation = pd.DataFrame(
        representation.groupby(level=[0, 1]).apply(lambda x: x.astype(str).apply("±".join, 0)))

    return means, stds, representation


def selection_by_sample_count(n_simulations, parent_intensities, region_limits, sim_param, alg_param,
                              max_sample_counts):
    """
    Receive two arrays representing the mean and std of the following measures:
    FDR - False discovery rate: The amount of false positives relative to the positives (FP & TP, i.e. large clusters);
    TPR - True positive rate: The amount of true positives relative to the condition positive (i.e. merged clusters);

    Large clusters are determined via the sample count.

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
    max_sample_counts : list
        Contains threshold values for sample counts.

    Returns
    -------
    means : numpy.ndarray
        Contains the means of false and true positives.
        Shape (parent_intensities, max_sample_counts, (false positives, true positives)).
    stds : numpy.ndarray
        Contains the stds of false and true positives.
        Shape (parent_intensities, max_sample_counts, (false positives, true positives)).
    """
    array = np.empty((n_simulations, len(parent_intensities), len(max_sample_counts), 2))
    for i in range(n_simulations):
        for j, (parent_intensity, limit) in enumerate(zip(parent_intensities, region_limits)):
            samples, labels, _ = si.sim_dstorm(parent_intensity=parent_intensity, lower_limit=-limit, upper_limit=limit,
                                               **sim_param)
            clust_labels = DBSCAN(**alg_param).fit_predict(samples)
            unique_labels, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
            merg_cluster_count, merg_cluster = cp.merged_clusters(clust_labels, labels)

            for h, max_sample_count in enumerate(max_sample_counts):
                large_cluster = unique_labels[np.where(sample_count > max_sample_count)[0]]
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


def visualization_selection(parent_intensity, sim_param, alg_param, max_area, n=None):
    """
    Receive the area of the convex hull and the sample count of the following cluster subsets:
    Simulation - the clusters simulated.
    Algorithm - the clusters found by the algorithm.
    Post-selection - the clusters of the algorithm not selected by large_clusters.
    Merged - the clusters of the algorithm identified as merged clusters.

    Parameters
    ----------
    parent_intensity : float
        The expected intensity of parent points per unit region measure.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_dstorm except parent_intensity.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN.
    max_area : numpy.ndarray
        Contains threshold areas, can be generated using lookup_tables.
    n : int, default=None
        If provided, n serves as a sample_count threshold, additionally to max_areas threshold.

    Returns
    -------
    simulation : numpy.ndarray
        Contains the area of the convex hull and the sample count of the clusters of the simulation.
    algorithm : numpy.ndarray
        Contains the area of the convex hull and the sample count of the clusters of the algorithm.
    post_selection : numpy.ndarray
        Contains the area of the convex hull and the sample count of the clusters of the algorithm not selected by
        large_clusters.
    merged_indices : numpy.ndarray
        Contains the area of the convex hull and the sample count of the clusters of the algorithm identified as
        merged clusters.
    """
    samples, labels, _ = si.sim_dstorm(parent_intensity=parent_intensity, **sim_param)
    _, sample_count_sim, area_chs_sim, _ = cp.cluster_property(labels, samples)
    clust_labels = DBSCAN(**alg_param).fit_predict(samples)
    unique_labels, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
    _, large_cluster = cp.large_clusters(unique_labels, sample_count, area_chs, max_area, n)
    _, merg_cluster = cp.merged_clusters(clust_labels, labels)
    merged_indices = np.where(np.in1d(unique_labels, merg_cluster))[0]
    leftover_indices = np.where(np.in1d(unique_labels, large_cluster, invert=True))[0]

    simulation = np.concatenate(([sample_count_sim], [area_chs_sim]))
    algorithm = np.concatenate(([sample_count], [area_chs]))
    post_selection = np.concatenate(([sample_count[leftover_indices]], [area_chs[leftover_indices]]))
    merged = np.concatenate(([sample_count[merged_indices]], [area_chs[merged_indices]]))

    return simulation, algorithm, post_selection, merged


def merged_cluster_sizes_comparison(n_simulations, sim_param, alg_param):
    """
    Receive lists of sample counts of clusters representing 1. merged clusters and 2. non-merged clusters.
    Therefore, the clustering result is used.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_dstorm.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN.

    Returns
    -------
    merged_clust_size_ : list
        Contains the different amounts of samples displayed by merged clusters.
    non_merg_clust_size_ : list
        Contains the different amounts of samples displayed by non-merged clusters.
    """
    merged_clust_size_ = []
    non_merg_clust_size_ = []
    for h in range(n_simulations):
        samples, sim_labels, _ = si.sim_dstorm(**sim_param)
        clust_labels = DBSCAN(**alg_param).fit_predict(samples)

        _, clust_size, _, _ = cp.cluster_property(clust_labels, samples)
        _, merged_labels = cp.merged_clusters(clust_labels, sim_labels)
        merged_clust_size = clust_size[merged_labels]
        non_merge_clust_size = np.delete(clust_size, merged_labels)
        merged_clust_size_.extend(merged_clust_size)
        non_merg_clust_size_.extend(non_merge_clust_size)

    return merged_clust_size_, non_merg_clust_size_