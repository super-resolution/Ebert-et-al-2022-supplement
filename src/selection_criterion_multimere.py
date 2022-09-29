import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import src.simulations as si
import src.cluster_properties as cp


def selection_criterion_analysis_multimeres(n_simulations, radii, sim_param, alg_param, max_areas, max_areas_keys,
                                            n=None):
    """
    Receive three tables representing the mean, std and mean±std of the following measures:
    Simulation multimeres - The relative amount of multimeres created by the simulation with respect to all created
    clusters;
    Algorithm multimeres - The relative amount of corresponding labels of the algorithm to simulated multimeres with
    respect to all clusters identified;
    TPR - True positive rate: The amount of true positives relative to the condition positive (i.e. multimeres);
    FDR - False discovery rate: The amount of false positives relative to the positives (FP & TP, i.e. large clusters);
    Algorithm large clusters - The relative amount of identified large clusters of the algorithm;
    Multimeric subdivision score simulation - measures subdivision rate of multimeres. 0 is no subdivision, 1 is
    complete subdivision. Uses the clusters of the ground truth. Example: 1 of 2 clusters is subdivided leading to a
    score of 0.5;
    Multimeric subdivision score algorithm - measures subdivision rate of multimeres. 0 is no subdivision, 1 is
    complete subdivision. Uses the clusters of the algorithm. Example (same as above): 2 of 3 clusters are a result of
    subdivision leading to a score of 0.66.

    Large clusters are determined via the convex hull.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    radii : list
        Contains all radius values (of simulations.sim_clusters) to be analyzed.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_clusters except radius.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN
    max_areas : list
        Contains numpy.ndarrays representing threshold areas. Each entry can be generated using lookup_tables.
    max_areas_keys : list
        Contains names of threshold arrays.
    n : int, default=None
        If provided, n serves as a sample_count threshold, additionally to max_areas threshold.

    Returns
    -------
    means : pandas.core.frame.DataFrame
        Contains the means of each measure for each radius (level 0) and threshold (level 1) combination.
    stds : pandas.core.frame.DataFrame
        Contains the stds of each measure for each radius (level 0) and threshold
        (level 1) combination.
    representation : pandas.core.frame.DataFrame
        Combination of means and stds for visual purposes.
    """
    list_3 = []
    for i in range(n_simulations):
        list_2 = []
        for radius in radii:
            samples, labels, original_size, _, _ = si.sim_clusters(radius=radius, **sim_param)
            unique_sim_labels = np.unique(np.delete(labels, np.where(labels == -1)))
            multimere_indices = np.where(original_size > 1)
            multimeres = unique_sim_labels[multimere_indices]
            # receive an index applicable to clust_labels
            multimeres_universal_index = np.where(np.in1d(labels, multimeres))[0]

            clust_labels = DBSCAN(**alg_param).fit_predict(samples)
            unique_labels, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
            multimeres_clust_labels = clust_labels[multimeres_universal_index]
            multimeres_clust_labels = np.unique(np.delete(multimeres_clust_labels,
                                                          np.where(multimeres_clust_labels == -1)))
            # the labels of the algorithm corresponding to multimeres;
            # there is not yet taken into account that these labels can be a result of subdivision (see below);
            # if a label of multimeres_clust_labels represents a result of merging, it is very likely that at least 1
            # multimere is part of the cluster, which makes it a target for the method;
            # it can be the case that a multimere of the ground truth is completely missed out by the algorithm
            # (label = -1),this scenario is neglected;
            multimeres_sim_rel = len(multimeres) / len(unique_sim_labels)
            multimeres_alg_rel = len(multimeres_clust_labels) / len(unique_labels)

            # the following is to measure cluster subdivision of multimeres;
            # Note: this is not taken into account when true and false positives are estimated!
            _, subdiv_cluster = cp.subdivided_clusters(clust_labels, labels)
            # check if/which multimeres got subdivided by the algorithm
            subdiv_multimeres_indices = np.where(np.in1d(subdiv_cluster, multimeres))[0]
            subdiv_multimeres_labels = subdiv_cluster[subdiv_multimeres_indices]
            # receive an index applicable to clust_labels
            subdiv_multimeres_universal_index = np.where(np.in1d(labels, subdiv_multimeres_labels))[0]
            subdiv_multimeres_clust_labels = clust_labels[subdiv_multimeres_universal_index]
            subdiv_multimeres_clust_labels = np.unique(np.delete(subdiv_multimeres_clust_labels,
                                                                 np.where(subdiv_multimeres_clust_labels == -1)))
            multimeric_subdivision_score_algorithm = len(subdiv_multimeres_clust_labels) / len(multimeres_clust_labels)
            multimeric_subdivision_score_simulation = len(subdiv_cluster) / len(multimeres)

            list_1 = []
            for max_area in max_areas:
                large_cluster_score, large_cluster = cp.large_clusters(unique_labels, sample_count, area_chs,
                                                                       max_area, n)
                large_cluster_rel = len(large_cluster) / len(unique_labels)  # compare with ratio of sim_clusters
                true_positives = np.where(np.in1d(large_cluster, multimeres_clust_labels))[0]
                if large_cluster_score > 0:
                    false_discovery_rate = 1 - len(true_positives)/len(large_cluster)
                else:
                    false_discovery_rate = np.nan
                true_positive_rate = len(true_positives) / len(multimeres_clust_labels)

                columns = [multimeres_sim_rel, multimeres_alg_rel, true_positive_rate, false_discovery_rate,
                           large_cluster_rel, multimeric_subdivision_score_simulation,
                           multimeric_subdivision_score_algorithm]
                columnnames = ["Simulation multimeres", "Algorithm multimeres", "TPR", "FDR",
                               "Algorithm large clusters", "Multimere subdivision simulation",
                               "Multimere subdivision algorithm"]
                table_1 = pd.DataFrame([columns], columns=columnnames)
                list_1.append(table_1)
            table_2 = pd.concat(list_1, keys=max_areas_keys, names=["Threshold"])
            list_2.append(table_2)
        table_3 = pd.concat(list_2, keys=radii, names=["Radius"])
        list_3.append(table_3)
    table_4 = pd.concat(list_3, keys=np.arange(0, n_simulations + 1))
    grouped_by_simulations = table_4.groupby(level=[1, 2])
    means = grouped_by_simulations.mean()
    stds = grouped_by_simulations.std()
    representation = pd.concat([means, stds]).applymap("{:.2f}".format)
    representation = pd.DataFrame(
        representation.groupby(level=[0, 1]).apply(lambda x: x.astype(str).apply("±".join, 0)))

    return means, stds, representation


def selection_multimeres_by_sample_count(n_simulations, radii, sim_param, alg_param, max_sample_counts):
    """
    Receive two arrays representing the mean and std of the following measures:
    FDR - False discovery rate: The amount of false positives relative to the positives (FP & TP, i.e. large clusters);
    TPR - True positive rate: The amount of true positives relative to the condition positive (i.e. multimeres);

    Large clusters are determined via the sample count.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    radii : list
        Contains all radius values (of simulations.sim_clusters) to be analyzed.
    sim_param : dict
        Contains all necessary arguments of simulations.sim_clusters except radius.
    alg_param : dict
        Contains all necessary arguments of sklearn.cluster.DBSCAN
    max_sample_counts : list
        Contains threshold values for sample counts.

    Returns
    -------
    means : numpy.ndarray
        Contains the means of FDR and TPR.
        Shape (radii, max_sample_counts, (FDR, TPR)).
    stds : numpy.ndarray
        Contains the stds of FDR and TPR.
        Shape (radii, max_sample_counts, (FDR, TPR)).
    """
    array = np.empty((n_simulations, len(radii), len(max_sample_counts), 2))
    for i in range(n_simulations):
        for j, radius in enumerate(radii):
            samples, labels, original_size, _, _ = si.sim_clusters(radius=radius, **sim_param)
            unique_sim_labels = np.unique(np.delete(labels, np.where(labels == -1)))
            multimere_indices = np.where(original_size > 1)
            multimeres = unique_sim_labels[multimere_indices]
            # receive an index applicable to clust_labels
            multimeres_universal_index = np.where(np.in1d(labels, multimeres))[0]

            clust_labels = DBSCAN(**alg_param).fit_predict(samples)
            unique_labels, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
            multimeres_clust_labels = clust_labels[multimeres_universal_index]
            multimeres_clust_labels = np.unique(np.delete(multimeres_clust_labels,
                                                          np.where(multimeres_clust_labels == -1)))
            for h, max_sample_count in enumerate(max_sample_counts):
                large_cluster = unique_labels[np.where(sample_count > max_sample_count)[0]]
                true_positives = np.where(np.in1d(large_cluster, multimeres_clust_labels))[0]
                if len(large_cluster) != 0:
                    false_discovery_rate = 1 - len(true_positives)/len(large_cluster)
                else:
                    false_discovery_rate = np.nan
                true_positive_rate = len(true_positives) / len(multimeres_clust_labels)

                array[i, j, h, 0] = false_discovery_rate
                array[i, j, h, 1] = true_positive_rate
    means = np.nanmean(array, axis=0)
    stds = np.nanstd(array, axis=0)

    return means, stds


def visualization_selection_multimeres(radius, sim_param, alg_param, max_area, n=None):
    """
    Receive the area of the convex hull and the sample count of the following cluster subsets:
    Simulation - the clusters simulated.
    Algorithm - the clusters found by the algorithm.
    Post-selection - the clusters of the algorithm not selected by large_clusters.
    Multimeres - the clusters of the algorithm corresponding to simulated multimeres.

    Parameters
    ----------
    radius : float
        Radius of circular area in case mode is "poisson" or standard deviation in case mode is "normal".
    sim_param : dict
        Contains all necessary arguments of simulations.sim_clusters except radius.
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
    multimeres : numpy.ndarray
        Contains the area of the convex hull and the sample count of the clusters of the algorithm corresponding to
        simulated multimeres.
    """
    samples, labels, original_size, _, _ = si.sim_clusters(radius=radius, **sim_param)
    unique_labels_sim, sample_count_sim, area_chs_sim, _ = cp.cluster_property(labels, samples)
    multimere_indices = np.where(original_size > 1)
    multimeres = unique_labels_sim[multimere_indices]
    multimeres_universal_index = np.where(np.in1d(labels, multimeres))[0]

    clust_labels = DBSCAN(**alg_param).fit_predict(samples)
    unique_labels, sample_count, area_chs, _ = cp.cluster_property(clust_labels, samples)
    _, large_cluster = cp.large_clusters(unique_labels, sample_count, area_chs, max_area, n)
    leftover_indices = np.where(np.in1d(unique_labels, large_cluster, invert=True))[0]
    multimeres_clust_labels = clust_labels[multimeres_universal_index]
    multimeres_clust_labels = np.unique(np.delete(multimeres_clust_labels,
                                                  np.where(multimeres_clust_labels == -1)))
    multimeres_index_alg = np.where(np.in1d(unique_labels, multimeres_clust_labels))[0]

    simulation = np.concatenate(([sample_count_sim], [area_chs_sim]))
    algorithm = np.concatenate(([sample_count], [area_chs]))
    post_selection = np.concatenate(([sample_count[leftover_indices]], [area_chs[leftover_indices]]))
    multimeres = np.concatenate(([sample_count[multimeres_index_alg]], [area_chs[multimeres_index_alg]]))

    return simulation, algorithm, post_selection, multimeres


def multimeric_cluster_sizes_comparison(n_simulations, sim_param):
    """
    Receive lists of sample counts of clusters representing 1. multimers and 2. monomers.
    The ground truth (not clustered) is used.

    Parameters
    ----------
    n_simulations : int
        Amount of simulations.
    sim_param : dict
     Contains all necessary arguments of simulations.sim_clusters.

    Returns
    -------
    multi_clust_size_ : list
        Contains the different amounts of samples displayed by multimers.
    mono_clust_size_ : list
        Contains the different amounts of samples displayed by monomers.
    """
    multi_clust_size_ = []
    mono_clust_size_ = []
    for h in range(n_simulations):
        samples, sim_labels, original_size, _, _ = si.sim_clusters(**sim_param)
        _, sample_count_sim, _, _ = cp.cluster_property(sim_labels, samples)
        multimere_indices = np.where(original_size > 1)

        multi_clust_size = sample_count_sim[multimere_indices]
        mono_clust_size = np.delete(sample_count_sim, multimere_indices)
        multi_clust_size_.extend(multi_clust_size)
        mono_clust_size_.extend(mono_clust_size)

    return multi_clust_size_, mono_clust_size_
