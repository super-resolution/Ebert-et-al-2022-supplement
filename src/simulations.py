import numpy as np
import scipy.stats as stats
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import src.truncated_geometric_distribution as tgd


def points_(region):
    """
    Receive vertices of the rectangular polygon.

    Parameters
    ----------
    region : shapely.geometry.polygon.Polygon
        The polygon representing the rectangle.

    Returns
    -------
    points : numpy.ndarray
        The vertices of the polygon.
    """
    corner = (region.bounds[0], region.bounds[1])
    width = region.bounds[2] - region.bounds[0]
    height = region.bounds[3] - region.bounds[1]
    rectangle = mpatches.Rectangle(corner, width, height, angle=0)
    points = rectangle.get_verts()

    return points


def contains(region, samples):
    """
    Receive indices of samples contained inside the close rectangle.

    Parameters
    ----------
    region : shapely.geometry.polygon.Polygon
        The polygon representing the rectangle.
    samples : numpy.ndarray
        The samples.

    Returns
    -------
    inside_indices : numpy.ndarray
        The indices of samples contained in the closed rectangle.
    """
    points = points_(region)
    polygon_path = mpath.Path(points, closed=True)
    mask = polygon_path.contains_points(samples)
    inside_indices = np.nonzero(mask)[0]

    return inside_indices


def make_poisson(intensity, region, seed=None):
    """
    Simulate random samples located in a rectangular region. The amount of samples is drawn from a poisson
    distribution with a rate of intensity * A(region).

    Parameters
    ----------
    intensity : float
        The expected intensity of samples per unit region measure.
    region : shapely.geometry.polygon.Polygon
        The polygon representing the rectangle.
    seed : numpy.random.BitGenerator, int or numpy.ndarray
        Seed of random number generation.

    Returns
    -------
    samples : numpy.ndarray
        The simulated samples.
    """
    rng = np.random.default_rng(seed)
    n_samples = rng.poisson(lam=intensity*region.area)
    corner = (region.bounds[0], region.bounds[1])
    width = region.bounds[2] - region.bounds[0]
    height = region.bounds[3] - region.bounds[1]
    new_samples = rng.random(size=(n_samples, 2))
    samples = (width, height) * new_samples + corner

    return samples


def random_circular(n_samples, centre, radius, seed):
    """
    Generate n_samples located randomly in a circular region with centre.

    Parameters
    ----------
    n_samples : int
        Amount of samples to be generated.
    centre : numpy.ndarray
        x and y of the centre.
    radius : float
        Radius of the circular region.
    seed : numpy.random.BitGenerator, int or numpy.ndarray
        Seed of random number generation.

    Returns
    -------
    samples : numpy.ndarray
        The generated samples.
    """
    rng = np.random.default_rng(seed)
    theta = rng.random(n_samples) * 2 * np.pi
    rho = radius * np.sqrt(rng.random(n_samples))

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    samples = np.array((x, y)).T + centre

    return samples


def limits(parent_intensities, parent_intensity, limit):
    """
    Receive a limit for each of parent_intensities, adjusted such that each product equals the product of the provided
    parent_intensity and limit.

    Parameters
    ----------
    parent_intensities : array-like
        The parent intensities whose limits shall be adjusted.
    parent_intensity : float
        The parent intensity whose limit is provided.
    limit : float
        The limit of the parent intensity.

    Returns
    -------
    all_limits : list
        A list of limits where each limit corresponds to an entry of parent_intensities.
    """
    all_limits = []
    for parent_int in parent_intensities:
        limit_ = np.sqrt(parent_intensity/parent_int * limit**2)
        all_limits.append(limit_)
    return all_limits


def make_dstorm(parent_intensity, lower_limit, upper_limit, cluster_mu, cluster_std, expansion_factor=6,
                min_points=0, clip=True, shuffle=True, seed=None):
    """
    Simulate samples distributed by complete spatial randomness (Poisson point process). These samples are
    the parent points. They are then used to produce offsprings normal distributed around the parent point.
    The amount of offsprings of each parent point is drawn from a geometric distribution.

    Parameters
    ----------
    parent_intensity : float
        The expected intensity of parent points per unit region measure.
    lower_limit : float
        Lower limit of x and y of the rectangular region.
    upper_limit : float
        Upper limit of x and y of the rectangular region.
    cluster_mu : float
        The mean of the geometric distribution of the offspring counts.
    cluster_std : float
        The standard deviation of the normal distribution with the parent points as mean.
    expansion_factor : float
        Factor by which the cluster_std is multiplied to set a distance by which the rectangular region is extended.
    min_points : int
        Defines the minimum x-value of the geometric distribution of the offspring counts.
    clip : bool
        If True, the result will be clipped to the non-extended rectangular region.
    shuffle : bool
        If True, the result is shuffled.
    seed : numpy.random.BitGenerator, int or numpy.ndarray
        Seed of random number generation.

    Returns
    -------
    samples : numpy.ndarray
        The offsprings generated by the parent points.
    labels : numpy.ndarray
        The labels of the offsprings. All offsprings originating from the same parent point share a unique label.
    parent_samples : numpy.ndarray
        The parent points.
    """
    rng = np.random.default_rng(seed)
    polygon = Polygon([(lower_limit, lower_limit), (lower_limit, upper_limit),
                       (upper_limit, upper_limit), (upper_limit, lower_limit)])

    # expand region
    expansion_distance = expansion_factor * np.max(cluster_std)
    polygon_expanded = polygon.buffer(expansion_distance)

    parent_samples = make_poisson(intensity=parent_intensity, region=polygon_expanded, seed=rng)
    n_cluster = len(parent_samples)

    n_offspring_list = rng.geometric(p=1 / (cluster_mu + 1 - min_points), size=n_cluster) - 1 + min_points
    cluster_std_ = np.full(shape=(n_cluster, 2), fill_value=cluster_std)
    samples = []
    labels = []
    for i, (parent, std, n_offspring) in enumerate(zip(parent_samples, cluster_std_, n_offspring_list)):
        offspring_samples = rng.normal(loc=parent, scale=std, size=(n_offspring, 2))
        samples.append(offspring_samples)
        labels += [i] * len(offspring_samples)

    samples = np.concatenate(samples) if len(samples) != 0 else np.array([])
    labels = np.array(labels)

    if clip is True:
        if len(samples) != 0:
            inside_indices = contains(polygon, samples)
            samples = samples[inside_indices]
            labels = labels[inside_indices]

    if shuffle:
        shuffled_indices = rng.permutation(len(samples))
        samples = samples[shuffled_indices]
        labels = labels[shuffled_indices]

    if len(samples) == 0:  # this is to convert empty arrays into arrays with shape (n_samples, n_features).
        samples = np.array([])
        samples = samples[:, np.newaxis]

    return samples, labels, parent_samples


def sim_dstorm(parent_intensity, lower_limit, upper_limit, cluster_mu, cluster_std, seed, min_samples):
    """
    Extends the function make_dstorm such that parent samples that were generated in the extended rectangular region
    are discarded. It also converts all labels, that occur less than min_samples times, into -1 (i.e. noise, many
    python implemented clustering algorithms use -1 as noise). Therefore, it generates a ground truth for potential
    clustering analysis.

    Parameters
    ----------
    parent_intensity : float
        The expected intensity of parent points per unit region measure.
    lower_limit : float
        Lower limit of x and y of the rectangular region.
    upper_limit : float
        Upper limit of x and y of the rectangular region.
    cluster_mu : float
        The mean of the geometric distribution of the offspring counts.
    cluster_std : float
        The standard deviation of the normal distribution with the parent points as mean.
    seed : numpy.random.BitGenerator, int or numpy.ndarray
        Seed of random number generation.
    min_samples : int
        Defines a minimum of occurrences of a unique label to be not converted to -1 (noise).

    Returns
    -------
    samples : numpy.ndarray
        The offsprings generated by the parent points.
    labels : numpy.ndarray
        The labels of the offsprings. If a label occurred less than min_samples times, it is -1.
    parent_samples_adj : numpy.ndarray
        The parent points, that are contained in the non-extended rectangular region.
    """
    samples, labels, parent_samples = make_dstorm(parent_intensity=parent_intensity, lower_limit=lower_limit,
                                                  upper_limit=upper_limit, cluster_mu=cluster_mu,
                                                  cluster_std=cluster_std, clip=True, seed=seed)
    polygon = Polygon([(lower_limit, lower_limit), (lower_limit, upper_limit),
                       (upper_limit, upper_limit), (upper_limit, lower_limit)])
    parent_samples_adj = []
    for i, parent_sample in enumerate(parent_samples):
        point = Point(parent_sample)
        if np.count_nonzero(labels == i) > min_samples - 1:
            parent_samples_adj.append(parent_sample)
        elif polygon.contains(point):
            parent_samples_adj.append(parent_sample)
            labels[labels == i] = -1
        else:
            labels[labels == i] = -1
    parent_samples_adj = np.array(parent_samples_adj)

    return samples, labels, parent_samples_adj


def make_clusters(parent_intensity, lower_limit, upper_limit, cluster_mu, cluster_std, ratio, radius, mode, p_value,
                  replace_min, replace_max, expansion_factor=6, min_points=0, clip=True, shuffle=True, seed=None):
    """
    Simulate samples distributed by complete spatial randomness (Poisson point process). These samples are the parent
    points. A certain amount (ratio) of them is selected and replaced by new parent points. The amount of new parents
    is drawn from a truncated geometric distribution. New parents are then randomly distributed around the original
    parents coordinates within a circular area of radius, if mode is "random". If mode is "normal", new
    parents are normal distributed around the original parent with the standard deviation radius.
    The non-selected parents and the new parents are then used to produce offsprings normal distributed around the
    parent point. The amount of offsprings of each parent point is drawn from a geometric distribution.

    Parameters
    ----------
    parent_intensity : float
        The expected intensity of parent points per unit region measure.
    lower_limit : float
        Lower limit of x and y of the rectangular region.
    upper_limit : float
        Upper limit of x and y of the rectangular region.
    cluster_mu : float
        The mean of the geometric distribution of the offspring counts.
    cluster_std : float
        The standard deviation of the normal distribution with the parent points as mean.
    ratio : float
        Number between 0 and 1 to determine the portion of replaced parent points.
    radius : float
        Radius of circular area in case mode is "poisson" or standard deviation in case mode is "normal".
    mode : str
        One of "random", "normal".
    p_value : float
        p of the geometric distribution of the new parent_counts.
    replace_min : int
        Defines the minimum x-value of the geometric distribution of the new parent_counts.
    replace_max : int
        Defines the maximum x_value of the geometric distribution of the new parent_counts.
    expansion_factor : float
        Factor by which the cluster_std is multiplied to set a distance by which the rectangular region is extended.
    min_points : int
        Defines the minimum x-value of the geometric distribution of the offspring counts.
    clip : bool
        If True, the result will be clipped to the non-extended rectangular region.
    shuffle : bool
        If True, the result is shuffled.
    seed : numpy.random.BitGenerator, int or numpy.ndarray
        Seed of random number generation.

    Returns
    -------
    samples : numpy.ndarray
        The offsprings generated by the parent points.
    labels : numpy.ndarray
        The labels of the offsprings. All offsprings originating from the same parent point share a unique label.
    parent_samples : numpy.ndarray
        The parent points.
    clustered_parents : list
        Arrays for each selected original parent point containing the indices of parent_samples representing their
        replacements.
    """
    rng = np.random.default_rng(seed)
    polygon = Polygon([(lower_limit, lower_limit), (lower_limit, upper_limit),
                       (upper_limit, upper_limit), (upper_limit, lower_limit)])

    expansion_distance = expansion_factor * np.max(cluster_std)
    polygon_expanded = polygon.buffer(expansion_distance)

    parent_samples = make_poisson(intensity=parent_intensity, region=polygon_expanded, seed=rng)
    selected_parents = np.array(rng.choice(parent_samples, size=int(len(parent_samples) * ratio),
                                           replace=False))
    indices = np.unique(np.where(np.isin(parent_samples, selected_parents))[0])
    parent_samples = np.delete(parent_samples, indices, axis=0)
    trunc_geo = tgd.truncated_geometric(p_value, replace_max, replace_min-1, replace_min)
    n_parent_replace = trunc_geo.rvs(size=len(selected_parents), random_state=rng)
    replace_mu = trunc_geo.moment(1)
    clustered_parents = []
    for i, (parent, replace) in enumerate(zip(selected_parents, n_parent_replace)):
        if mode == "random":
            new_parents = random_circular(replace, parent, radius, rng)
        elif mode == "normal":
            new_parents = rng.normal(loc=parent, scale=radius, size=(replace, 2))
        else:
            new_parents = None
        parent_samples = np.concatenate((parent_samples, new_parents))
        clustered_parents_ = np.unique(np.where(np.isin(parent_samples, new_parents))[0])
        clustered_parents.append(clustered_parents_)
    n_cluster = len(parent_samples)

    n_offspring_list = rng.geometric(p=1 / (cluster_mu + 1 - min_points), size=n_cluster) - 1 + min_points
    cluster_std_ = np.full(shape=(n_cluster, 2), fill_value=cluster_std)
    samples = []
    labels = []
    for i, (parent, std, n_offspring) in enumerate(zip(parent_samples, cluster_std_, n_offspring_list)):
        offspring_samples = rng.normal(loc=parent, scale=std, size=(n_offspring, 2))
        samples.append(offspring_samples)
        labels += [i] * len(offspring_samples)

    samples = np.concatenate(samples) if len(samples) != 0 else np.array([])
    labels = np.array(labels)

    if clip is True:
        if len(samples) != 0:
            inside_indices = contains(polygon, samples)
            samples = samples[inside_indices]
            labels = labels[inside_indices]

    if shuffle:
        shuffled_indices = rng.permutation(len(samples))
        samples = samples[shuffled_indices]
        labels = labels[shuffled_indices]

    if len(samples) == 0:  # this is to convert empty arrays into arrays with shape (n_samples, n_features).
        samples = np.array([])
        samples = samples[:, np.newaxis]

    return samples, labels, parent_samples, clustered_parents, replace_mu


def sim_clusters(parent_intensity, lower_limit, upper_limit, cluster_mu, cluster_std, ratio, radius,
                 mode, p_value, replace_min, replace_max, seed, min_samples):
    """
    Extends the function make_clusters such that parent samples that were generated in the extended rectangular
    region are discarded. All labels of samples that originate from parents replacing one original parent are unified
    (new labels, also contains unaffected labels). It then converts all new labels, that occur less than min_samples
    times, into -1 (i.e. noise, many python implemented clustering algorithms use -1 as noise). The amount of parents
    behind each kept new label is also stored.

    Parameters
    ----------
    parent_intensity : float
        The expected intensity of parent points per unit region measure.
    lower_limit : float
        Lower limit of x and y of the rectangular region.
    upper_limit : float
        Upper limit of x and y of the rectangular region.
    cluster_mu : float
        The mean of the geometric distribution of the offspring counts.
    cluster_std : float
        The standard deviation of the normal distribution with the parent points as mean.
    ratio : float
        Number between 0 and 1 to determine the portion of replaced parent points.
    radius : float
        Radius of circular area in case mode is "poisson" or standard deviation in case mode is "normal".
    p_value : float
        p of the geometric distribution of the new parent counts.
    mode : str
        One of "poisson", "normal".
    replace_min : int
        Defines the minimum x-value of the geometric distribution of the new parent_counts.
    replace_max : int
        Defines the maximum x_value of the geometric distribution of the new parent_counts.
    seed : numpy.random.BitGenerator, int or numpy.ndarray
        Seed of random number generation.
    min_samples : int
        Defines a minimum of occurrences of a unique (new) label to be not converted to -1 (noise).

    Returns
    -------
    samples : numpy.ndarray
        The offsprings generated by the parent points.
    labels : numpy.ndarray
        The new labels of the offsprings. If a new label occurred less than min_samples times, it is -1.
    original_size : numpy.ndarray
        The original amount of labels a new label is representing.
    parent_samples_adj : numpy.ndarray
        The parent points, that are contained in the non-extended rectangular region.
    """
    samples, labels, parent_samples, clustered_parents, replace_mu = \
        make_clusters(parent_intensity=parent_intensity, lower_limit=lower_limit,
                      upper_limit=upper_limit, cluster_mu=cluster_mu, cluster_std=cluster_std,
                      seed=seed, mode=mode, ratio=ratio, radius=radius, p_value=p_value,
                      replace_min=replace_min, replace_max=replace_max)
    new_labels = np.copy(labels)

    for clustered_parents_ in clustered_parents:
        if len(clustered_parents_) > 0:
            right_label = clustered_parents_[0]
            wrong_label = clustered_parents_[1:]
            wrong_indices = np.in1d(new_labels, wrong_label).nonzero()  # nonzero on boolean values equals True
            new_labels[wrong_indices] = right_label

    polygon = Polygon([(lower_limit, lower_limit), (lower_limit, upper_limit),
                       (upper_limit, upper_limit), (upper_limit, lower_limit)])
    parent_samples_adj = []
    for i, parent_sample in enumerate(parent_samples):
        point = Point(parent_sample)
        indices = np.where(labels == i)
        new_label = np.unique(new_labels[indices])
        if np.count_nonzero(new_labels == new_label) > min_samples - 1:
            parent_samples_adj.append(parent_sample)
        elif polygon.contains(point):
            parent_samples_adj.append(parent_sample)
            new_labels[indices] = -1
        else:
            new_labels[indices] = -1
    parent_samples_adj = np.array(parent_samples_adj)

    uniques = np.unique(new_labels)
    uniques = np.delete(uniques, np.where(uniques == -1))
    original_size = []
    for i in uniques:
        indices = np.where(new_labels == i)
        corres_labels = labels[indices]
        cor_labl_len = len(np.unique(corres_labels))
        original_size.append(cor_labl_len)
    original_size = np.array(original_size)

    return samples, new_labels, original_size, parent_samples_adj, replace_mu


######################################################################################################################
# varying intensities / photon counts per localization lead to varying cluster_stds
######################################################################################################################
def fwhm_to_std(fwhm):
    """
    Receive the standard deviation of the point spread function given the full width at half maximum.

    Parameters
    ----------
    fwhm : float
        The full width at half maximum of the point spread function.

    Returns
    -------
    std : float
        The standard deviation of the point spread function.
    """
    std = fwhm / (2 * np.sqrt(2 * np.log(2)))

    return std


def intensity_to_photons(intensities, factor):
    """
    Intensity values measured may not represent photon counts. To convert them, the digital conversion factor has to be
    known.

    Parameters
    ----------
    intensities : float or numpy.ndarray
        The intensity values measured.
    factor : float
        The digital conversion factor used to receive the intensities.

    Returns
    -------
    photons : float or numpy.ndarray
        The underlying photon counts of the measured intensity values.
    """
    photons = intensities * (1/factor)

    return photons


def compute_cluster_std(psf_std, pixel_size, photons, background, emccd):
    """
    The localization precision is computed as described by S. Stallinga and B. Rieger (Visualization and resolution
    in localization microscopy).

    Parameters
    ----------
    psf_std : float
        Standard deviation of the Gaussian used to fit the point spread function.
    pixel_size : float
        The pixel size.
    photons : float or numpy.ndarray
        The amount of photons of the localization to calculate its precision.
        Physically, this would be of type int. In measurements, intensity values may represent amplified photon counts.
        Their back-calculated results are of type float.
    background : float
        The amount of background photons per pixel.
    emccd : bool
        If an emccd camera was used or shall be reproduced, a factor of sqrt(2) is included (because of excess noise).

    Returns
    -------
    cluster_std : float
        The standard deviation of normal distributed offsprings / samples. In single molecule localization microscopy,
        also called localization precision.
    """
    std_a_squared = psf_std**2 + pixel_size**2 / 12
    tau = (2 * np.pi * std_a_squared * background) / (photons * pixel_size**2)
    cluster_std_squared = std_a_squared/photons * (1 + 4*tau + np.sqrt((2*tau) / (1 + 4*tau)))
    cluster_std = np.sqrt(cluster_std_squared)
    if emccd:
        cluster_std = np.sqrt(2) * cluster_std

    return cluster_std


def make_std_vary(parent_intensity, lower_limit, upper_limit, cluster_mu, photons_min, photons_mean, photons_background,
                  psf_std, pixel_size, emccd=True, gamma_a=1.1, expansion_factor=6, min_points=0, clip=True,
                  shuffle=True, seed=None):
    """
    Simulate samples distributed by complete spatial randomness (Poisson point process). These samples are the parent
    points. They are then used to produce offsprings normal distributed around the parent point. The standard deviation
    of this normal distribution of each offspring depends on the photon count drawn from a gamma distribution. The
    gamma distribution is of shape alpha=gamma_a and scale=photons_mean/gamma_a.
    The amount of offsprings of each parent point is drawn from a geometric distribution.

    Parameters
    ----------
    parent_intensity : float
        The expected intensity of parent points per unit region measure.
    lower_limit : float
        Lower limit of x and y of the rectangular region.
    upper_limit : float
        Upper limit of x and y of the rectangular region.
    cluster_mu : float
        The mean of the geometric distribution of the offspring counts.
    photons_min : float
        The minimum photon count as a threshold to be applied to drawn samples from the geometric distribution.
    photons_mean : float
        The mean of photon counts (taken a distribution that starts at values of 0), resembling the mean of the gamma
        distribution. It is connected to the second shape parameter of the gamma distribution as parameterized in
        scipy.stats.gamma, that is the scale parameter: scale = mean/gamma_a
    photons_background : float
        The background photon count per pixel.
    psf_std : float
        The standard deviation of the point spread function.
    pixel_size : float
        The pixel size.
    emccd : bool
        If an emccd camera was used or shall be reproduced, a factor of sqrt(2) is included (because of excess noise).
    gamma_a : float, default=1.1
        The first shape parameter of the gamma distribution as parameterized in scipy.stats.gamma. 1 would resemble
        the exponential distribution.
    expansion_factor : float
        Factor by which the cluster_std is multiplied to set a distance by which the rectangular region is extended.
    min_points : int
        Defines the minimum x-value of the geometric distribution of the offspring counts.
    clip : bool
        If True, the result will be clipped to the non-extended rectangular region.
    shuffle : bool
        If True, the result is shuffled.
    seed : numpy.random.BitGenerator, int or numpy.ndarray
        Seed of random number generation.

    Returns
    -------
    samples : numpy.ndarray
        The offsprings generated by the parent points.
    labels : numpy.ndarray
        The labels of the offsprings. All offsprings originating from the same parent point share a unique label.
    parent_samples : numpy.ndarray
        The parent points.
    mean_cluster_std : float
        The mean of the varying standard deviation of the normal distributed offsprings.
    cluster_stds : numpy.ndarray
        The collection of standard deviation values of all offsprings.
    photons : numpy.ndarray
        The collection of photon counts of all offsprings.
    """
    rng = np.random.default_rng(seed)
    polygon = Polygon([(lower_limit, lower_limit), (lower_limit, upper_limit),
                       (upper_limit, upper_limit), (upper_limit, lower_limit)])

    cluster_std_max = compute_cluster_std(psf_std, pixel_size, photons_min, photons_background, emccd)
    expansion_distance = expansion_factor * cluster_std_max
    polygon_expanded = polygon.buffer(expansion_distance)

    parent_samples = make_poisson(intensity=parent_intensity, region=polygon_expanded, seed=rng)
    n_cluster = len(parent_samples)
    n_offspring_list = rng.geometric(p=1 / (cluster_mu + 1 - min_points), size=n_cluster) - 1 + min_points

    photons = stats.gamma.rvs(a=gamma_a, loc=0, scale=photons_mean/gamma_a, size=10000000, random_state=rng)
    photons_trunc = photons[photons > photons_min]
    photons = [rng.choice(photons_trunc, size=n) for n in n_offspring_list]

    cluster_stds = [compute_cluster_std(psf_std=psf_std, pixel_size=pixel_size, photons=np.array(photons_),
                                        background=photons_background, emccd=emccd) for photons_ in photons]

    samples = []
    labels = []
    for i, (parent, std, n_offspring) in enumerate(zip(parent_samples, cluster_stds, n_offspring_list)):
        offspring_samples = [rng.normal(loc=parent, scale=std_, size=(1, 2)) for std_ in std]
        if len(offspring_samples) != 0:
            offspring_samples = np.vstack(offspring_samples)
        else:
            offspring_samples = np.empty((0, 2))
        samples.append(offspring_samples)
        labels += [i] * len(offspring_samples)
    cluster_stds = np.concatenate(cluster_stds)
    mean_cluster_std = np.mean(cluster_stds)
    photons = np.concatenate(photons)
    samples = np.concatenate(samples) if len(samples) != 0 else np.array([])
    labels = np.array(labels)

    if clip is True:
        if len(samples) != 0:
            inside_indices = contains(polygon, samples)
            samples = samples[inside_indices]
            labels = labels[inside_indices]

    if shuffle:
        shuffled_indices = rng.permutation(len(samples))
        samples = samples[shuffled_indices]
        labels = labels[shuffled_indices]

    if len(samples) == 0:  # this is to convert empty arrays into arrays with shape (n_samples, n_features).
        samples = np.array([])
        samples = samples[:, np.newaxis]

    return samples, labels, parent_samples, mean_cluster_std, cluster_stds, photons


def sim_std_vary(parent_intensity, lower_limit, upper_limit, cluster_mu, photons_min, photons_mean,
                 photons_background, psf_std, pixel_size, emccd, gamma_a, seed, min_samples):
    """
    Extends the function make_std_vary such that parent samples that were generated in the extended rectangular region
    are discarded. It also converts all labels, that occur less than min_samples times, into -1 (i.e. noise, many
    python implemented clustering algorithms use -1 as noise). Therefore, it generates a ground truth for potential
    clustering analysis.

    Parameters
    ----------
    parent_intensity : float
        The expected intensity of parent points per unit region measure.
    lower_limit : float
        Lower limit of x and y of the rectangular region.
    upper_limit : float
        Upper limit of x and y of the rectangular region.
    cluster_mu : float
        The mean of the geometric distribution of the offspring counts.
    photons_min : float
        The minimum photon count as a threshold to be applied to drawn samples from the geometric distribution.
    photons_mean : float
        The mean of photon counts (taken a distribution that starts at values of 0), resembling the mean of the gamma
        distribution. It is connected to the second shape parameter of the gamma distribution as parameterized in
        scipy.stats.gamma, that is the scale parameter: scale = mean/gamma_a
    photons_background : float
        The background photon count per pixel.
    psf_std : float
        The standard deviation of the point spread function.
    pixel_size : float
        The pixel size.
    emccd : bool
        If an emccd camera was used or shall be reproduced, a factor of sqrt(2) is included (because of excess noise).
    gamma_a : float, default=1.1
        The first shape parameter of the gamma distribution as parameterized in scipy.stats.gamma. 1 would resemble
        the exponential distribution.
    seed : numpy.random.BitGenerator, int or numpy.ndarray
        Seed of random number generation.
    min_samples : int
        Defines a minimum of occurrences of a unique (new) label to be not converted to -1 (noise).

    Returns
    -------
    samples : numpy.ndarray
        The offsprings generated by the parent points.
    labels : numpy.ndarray
        The labels of the offsprings. All offsprings originating from the same parent point share a unique label.
    parent_samples : numpy.ndarray
        The parent points.
    mean_cluster_std : float
        The mean of the varying standard deviation of the normal distributed offsprings.
    cluster_stds : numpy.ndarray
        The collection of standard deviation values of all offsprings.
    photons : numpy.ndarray
        The collection of photon counts of all offsprings.
    parent_samples_adj : numpy.ndarray
        The parent points, that are contained in the non-extended rectangular region.
    """
    samples, labels, parent_samples, mean_cluster_std, cluster_stds, photons = \
        make_std_vary(parent_intensity=parent_intensity, lower_limit=lower_limit, upper_limit=upper_limit,
                      cluster_mu=cluster_mu, photons_min=photons_min, photons_mean=photons_mean, seed=seed,
                      photons_background=photons_background, psf_std=psf_std, pixel_size=pixel_size, emccd=emccd,
                      gamma_a=gamma_a)

    polygon = Polygon([(lower_limit, lower_limit), (lower_limit, upper_limit),
                       (upper_limit, upper_limit), (upper_limit, lower_limit)])
    parent_samples_adj = []
    for i, parent_sample in enumerate(parent_samples):
        point = Point(parent_sample)
        if np.count_nonzero(labels == i) > min_samples - 1:
            parent_samples_adj.append(parent_sample)
        elif polygon.contains(point):
            parent_samples_adj.append(parent_sample)
            labels[labels == i] = -1
        else:
            labels[labels == i] = -1
    parent_samples_adj = np.array(parent_samples_adj)

    return samples, labels, parent_samples, mean_cluster_std, cluster_stds, photons, parent_samples_adj
