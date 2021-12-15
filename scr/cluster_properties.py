import numpy as np
from scipy.spatial import ConvexHull


def wrong_clusters(clust_labels, sim_labels):
    """
    Receive the (amount of) clusters of the algorithm that consist solely of samples declared as noise in the
    ground truth (i.e. wrong clusters).

    Parameters
    ----------
    clust_labels : numpy.ndarray
        Labels resulting from clustering.
    sim_labels : numpy.ndarray
        Labels of the ground truth.

    Returns
    -------
    wrong_cluster_count : int
        Wrong cluster count.
    wrong_cluster : numpy.ndarray
        Labels of wrong clusters (algorithm).
    """
    wrong_cluster_count = 0
    wrong_cluster = []
    for i in np.unique(clust_labels):
        if i == -1:
            continue
        elif np.all(sim_labels[np.argwhere(clust_labels == i)] == -1):
            wrong_cluster_count += 1
            wrong_cluster.append(i)
    wrong_cluster = np.array(wrong_cluster)

    return wrong_cluster_count, wrong_cluster


def missed_clusters(clust_labels, sim_labels):
    """
    Receive the (amount of) clusters of the ground truth that consist solely of samples declared as noise by the
    algorithm (i.e. missed clusters).
    Parameters
    ----------
    clust_labels : numpy.ndarray
        Labels resulting from clustering.
    sim_labels : numpy.ndarray
        Labels of the ground truth.

    Returns
    -------
    missed_cluster_count : int
        Missed cluster count.
    missed_cluster : numpy.ndarray
        Labels of missed clusters (ground truth).
    """
    missed_cluster_count = 0
    missed_cluster = []
    for i in np.unique(sim_labels):
        if i == -1:
            continue
        elif np.all(clust_labels[np.argwhere(sim_labels == i)] == -1):
            missed_cluster_count += 1
            missed_cluster.append(i)
    missed_cluster = np.array(missed_cluster)

    return missed_cluster_count, missed_cluster


def subdivided_clusters(clust_labels, sim_labels):
    """
    Receive the clusters of the ground truth that are affected by cluster subdivision, i.e. n labels that correspond
    to m labels in the algorithm, where n < m. Receive the difference in cluster counts between the ground truth and
    the algorithm caused by cluster subdivision.

    Parameters
    ----------
    clust_labels : numpy.ndarray
        Labels resulting from clustering.
    sim_labels : numpy.ndarray
        Labels of the ground truth.

    Returns
    -------
    subdiv_cluster_count : int
        Amount of clusters additionally found in the result of the algorithm due to subdivision.
    subdiv_cluster : numpy.ndarray
        Labels of subdivided clusters (ground truth).
    """
    subdiv_cluster_count = 0
    skip_label = []
    for i in np.unique(sim_labels):
        if i == -1:
            continue
        elif i not in skip_label:
            old_corres_simulation_labels = [i]
            indices_simulation = np.where(sim_labels == i)
            old_corres_cluster_labels = np.unique(clust_labels[indices_simulation])
            old_corres_cluster_labels = old_corres_cluster_labels[old_corres_cluster_labels > -1]
            if len(old_corres_cluster_labels) > 1:
                indices_cluster = np.where(np.in1d(clust_labels, old_corres_cluster_labels))[0]
                new_corres_simulation_labels = np.unique(sim_labels[indices_cluster])
                new_corres_simulation_labels = new_corres_simulation_labels[new_corres_simulation_labels > -1]
                new_indices_simulation = np.where(np.in1d(sim_labels, new_corres_simulation_labels))[0]
                new_corres_cluster_labels = np.unique(clust_labels[new_indices_simulation])
                new_corres_cluster_labels = new_corres_cluster_labels[new_corres_cluster_labels > -1]
                while len(old_corres_simulation_labels) < len(new_corres_simulation_labels) or \
                        len(old_corres_cluster_labels) < len(new_corres_cluster_labels):
                    old_corres_simulation_labels = new_corres_simulation_labels
                    old_corres_cluster_labels = new_corres_cluster_labels
                    indices_cluster = np.where(np.in1d(clust_labels, old_corres_cluster_labels))[0]
                    new_corres_simulation_labels = np.unique(sim_labels[indices_cluster])
                    new_corres_simulation_labels = new_corres_simulation_labels[new_corres_simulation_labels > -1]
                    new_indices_simulation = np.where(np.in1d(sim_labels, new_corres_simulation_labels))[0]
                    new_corres_cluster_labels = np.unique(clust_labels[new_indices_simulation])
                    new_corres_cluster_labels = new_corres_cluster_labels[new_corres_cluster_labels > -1]
                if len(new_corres_simulation_labels) < len(new_corres_cluster_labels):
                    skip_label += list(new_corres_simulation_labels)
                    subdiv_cluster_count += (len(new_corres_cluster_labels) - len(new_corres_simulation_labels))
    subdiv_cluster = np.array(skip_label)

    return subdiv_cluster_count, subdiv_cluster


def merged_clusters(clust_labels, sim_labels):
    """
    Receive the clusters of the algorithm that are affected by cluster merging, i.e. m labels that correspond
    to n labels in the ground truth, where m < n. Receive the difference in cluster counts between the ground truth and
    the algorithm caused by cluster merging.

    Parameters
    ----------
    clust_labels : numpy.ndarray
        Labels resulting from clustering.
    sim_labels : numpy.ndarray
        Labels of the ground truth.

    Returns
    -------
    merg_cluster_count : int
        Amount of clusters lost in the result of the algorithm due to merging.
    merg_cluster : numpy.ndarray
        Labels of merged clusters (algorithm).
    """
    merg_cluster_count = 0
    skip_label = []
    for i in np.unique(clust_labels):
        if i == -1:
            continue
        elif i not in skip_label:
            old_corres_cluster_labels = [i]
            indices_cluster = np.where(clust_labels == i)
            old_corres_simulation_labels = np.unique(sim_labels[indices_cluster])
            old_corres_simulation_labels = old_corres_simulation_labels[old_corres_simulation_labels > -1]
            if len(old_corres_simulation_labels) > 1:
                indices_simulation = np.where(np.in1d(sim_labels, old_corres_simulation_labels))[0]
                new_corres_cluster_labels = np.unique(clust_labels[indices_simulation])
                new_corres_cluster_labels = new_corres_cluster_labels[new_corres_cluster_labels > -1]
                new_indices_cluster = np.where(np.in1d(clust_labels, new_corres_cluster_labels))[0]
                new_corres_simulation_labels = np.unique(sim_labels[new_indices_cluster])
                new_corres_simulation_labels = new_corres_simulation_labels[new_corres_simulation_labels > -1]
                while len(old_corres_simulation_labels) < len(new_corres_simulation_labels) or \
                        len(old_corres_cluster_labels) < len(new_corres_cluster_labels):
                    old_corres_cluster_labels = new_corres_cluster_labels
                    old_corres_simulation_labels = new_corres_simulation_labels
                    indices_simulation = np.where(np.in1d(sim_labels, old_corres_simulation_labels))[0]
                    new_corres_cluster_labels = np.unique(clust_labels[indices_simulation])
                    new_corres_cluster_labels = new_corres_cluster_labels[new_corres_cluster_labels > -1]
                    new_indices_cluster = np.where(np.in1d(clust_labels, new_corres_cluster_labels))[0]
                    new_corres_simulation_labels = np.unique(sim_labels[new_indices_cluster])
                    new_corres_simulation_labels = new_corres_simulation_labels[new_corres_simulation_labels > -1]
                if len(new_corres_cluster_labels) < len(new_corres_simulation_labels):
                    skip_label += list(new_corres_cluster_labels)
                    merg_cluster_count += (len(new_corres_simulation_labels) - len(new_corres_cluster_labels))
    merg_cluster = np.array(skip_label)

    return merg_cluster_count, merg_cluster


def correct_clusters(clust_labels, wrong_cluster_count, missed_cluster_count, subdiv_cluster_count, merg_cluster_count):
    """
    Calculate the amount of clusters found by the algorithm that can be considered correct clusters. This includes
    subtracting the amount of wrong clusters and additional clusters due to subdivision. The result is then taken
    relative to all clusters found by the algorithm plus the amount of missed clusters and the amount of lost clusters
    due to merging.

    Parameters
    ----------
    clust_labels : numpy.ndarray
        Labels resulting from clustering.
    wrong_cluster_count : int
        Wrong cluster count.
    missed_cluster_count : int
        Missed cluster count.
    subdiv_cluster_count : int
        Amount of clusters additionally found in the result of the algorithm due to subdivision.
    merg_cluster_count : int
        Amount of clusters lost in the result of the algorithm due to merging.

    Returns
    -------
    correct_clust_score : float
        Number between 0 and 1. 0 indicates no correct clusters, 1 indicates that all clusters were correctly
        identified.
    """
    unique_labels = np.unique(clust_labels)
    delete_index = np.where(unique_labels == -1)
    unique_labels = np.delete(unique_labels, delete_index)
    clust_count = len(unique_labels)
    correct_clust = clust_count - wrong_cluster_count - subdiv_cluster_count
    expected_clust = clust_count + missed_cluster_count + merg_cluster_count
    correct_clust_score = correct_clust / expected_clust

    return correct_clust_score


def cluster_property(clust_labels, samples):
    """
    Receive several cluster properties: the unique labels, their associated sample counts, their associated areas of
    the convex hull as well as their center coordinates.

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
    area_chs : numpy.ndarray
        Contains the area of the convex hull of each of unique_labels.
    coordinates : numpy.ndarray
        Contains the center coordinate of each of unique_labels.
    """
    sample_count = []
    area_chs = []
    coordinates = []

    unique_labels = np.unique(clust_labels)
    unique_labels = np.delete(unique_labels, np.where(unique_labels == -1))
    for i in unique_labels:
        indices_cluster = np.where(clust_labels == i)
        samples_cluster = samples[indices_cluster]
        sample_count.append(len(samples_cluster))
        area_ch = ConvexHull(samples_cluster).volume
        area_chs.append(area_ch)
        coordinate = np.mean(samples_cluster, axis=0)
        coordinates.append(coordinate)

    sample_count = np.array(sample_count)
    area_chs = np.array(area_chs)
    coordinates = np.array(coordinates)

    return unique_labels, sample_count, area_chs, coordinates


def large_clusters(unique_labels, sample_count, area_chs, max_areas, n=None):
    """
    Receive the clusters with convex hulls larger than some threshold values (max_areas). Additionally, another
    threshold regarding the sample count can be applied (n).

    Parameters
    ----------
    unique_labels : numpy.ndarray
        First result of cluster_property.
    sample_count : numpy.ndarray
        Second result of cluster_property.
    area_chs : numpy.ndarray
        Third result of cluster_property.
    max_areas : numpy.ndarray
        Threshold areas. Can be generated using lookup_tables.
    n : int, default=None
        If provided, n serves as a sample_count threshold.

    Returns
    -------
    large_cluster_score : float
        The amount of large clusters relative to examined clusters.
    large_cluster : numpy.ndarray
        Labels of large clusters.
    """
    cluster_count = 0
    large_cluster_count = 0
    large_cluster = []

    if n:
        for i, num in enumerate(max_areas[n-3:]):
            indices = np.where(sample_count == i + n)[0]
            areas = area_chs[indices]
            large = np.where(areas > num)[0]
            idx = indices[large]
            labels = unique_labels[idx]
            large_cluster.append(labels)
            large_cluster_count += len(large)
            cluster_count += len(areas)
    else:
        for i, num in enumerate(max_areas):
            indices = np.where(sample_count == i + 3)[0]
            areas = area_chs[indices]
            large = np.where(areas > num)[0]
            idx = indices[large]
            labels = unique_labels[idx]
            large_cluster.append(labels)
            large_cluster_count += len(labels)
            cluster_count += len(areas)

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


def clustered_samples(clust_labels):
    """
    Receive the amount of samples contained in clusters.

    Parameters
    ----------
    clust_labels : numpy.ndarray
        Labels resulting from clustering.

    Returns
    -------
    clust_samples : int
        Amount of samples contained in clusters.
    """
    clust_samples = len(np.delete(clust_labels, np.where(clust_labels == -1)))

    return clust_samples
