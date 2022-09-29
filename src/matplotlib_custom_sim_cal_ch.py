import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm
from matplotlib import rcParams
from brokenaxes import brokenaxes
import src.convex_hull as ch
from src.matplotlib_custom import second_axis, second_axis_log


def plot_1(measure, n_points, y_1, yerr_high_1, yerr_low_1, y_2, yerr_high_2, yerr_low_2):
    """
    Generation of figure 1 of "simulation_calculation_convex_hull".

    Parameters
    ----------
    measure : str
        One of "peri", "area", "surf_area", "volume".
    n_points : numpy.ndarray
        x-values of the plot.
    y_1 : numpy.ndarray
        First y-values of the plot. The third normalized result of convex_hull.sided_deviations.
    yerr_high_1 : numpy.ndarray
        First positive y-error of the plot. The first normalized result of convex_hull.sided_deviations.
    yerr_low_1 : numpy.ndarray
        First negative y-error of the plot. The second normalized result of convex_hull.sided_deviations.
    y_2 : numpy.ndarray
        Second y-values of the plot. The third normalized result of convex_hull.sided_deviations.
    yerr_high_2 : numpy.ndarray
        Second positive y-error of the plot. The first normalized result of convex_hull.sided_deviations.
    yerr_low_2 : numpy.ndarray
        Second negative y-error of the plot. The second normalized result of convex_hull.sided_deviations.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure 1.
    bax : brokenaxes.BrokenAxes
        bax is returned so that it can be passed to customize_figure.
    """
    rcParams["axes.linewidth"] = 2
    fig = plt.figure(figsize=(6, 4))
    bax = brokenaxes(xlims=((0, 20.5), (89.5, 100.5)), hspace=.15)
    bax.errorbar(x=n_points, y=y_1, yerr=[yerr_low_1, yerr_high_1], capsize=3, fmt="x", lw=2, markersize=6,
                 markeredgewidth=1.5)
    bax.errorbar(x=n_points, y=y_2, yerr=[yerr_low_2, yerr_high_2], capsize=3, fmt="x", lw=2, markersize=6,
                 markeredgewidth=1.5)
    if measure == "peri":
        bax.plot(n_points, ch.calc_peri_convex_hull_2d_pt2(1, n_points), color="k", lw=2)
        bax.set_ylabel(r"$P_{ch}/\sigma$", fontsize=21)
    elif measure == "area":
        bax.plot(n_points, ch.calc_area_convex_hull_2d_pt2(1, n_points), color="k", lw=2)
        bax.set_ylabel(r"$A_{ch}/\sigma^2$", fontsize=21)
    elif measure == "surf_area":
        bax.plot(n_points, ch.calc_area_convex_hull_3d_pt2(1, 1, n_points), color="k", lw=2)
        bax.set_ylabel(r"$S_{ch}/factor$", fontsize=21)
    elif measure == "volume":
        bax.plot(n_points, ch.calc_volume_convex_hull_3d_pt2(1, 1, n_points), color="k", lw=2)
        bax.set_ylabel(r"$V_{ch}/(\sigma_{xy}^2*\sigma_z)$", fontsize=21)
    bax.set_xlabel(r"$n$", fontsize=21, labelpad=25)

    return fig, bax


def customize_figure(tick_spacing_y, tick_spacing_x, bax):
    """
    Customization of figure 1 of "simulation_calculation_convex_hull".

    Parameters
    ----------
    tick_spacing_y : float
        The base of matplotlib.ticker.MultipleLocator of the y-axis.
    tick_spacing_x : float
        The base of matplotlib.ticker.MultipleLocator of the x-axis.
    bax : brokenaxes.BrokenAxes
        Second result of plot_1.
    """
    d_kwargs = dict(transform=bax.fig.transFigure, color=bax.diag_color, clip_on=False,
                    lw=rcParams["axes.linewidth"])
    size = bax.fig.get_size_inches()
    ylen = bax.d * np.sin(bax.tilt * np.pi / 180) * size[0] / size[1]
    xlen = bax.d * np.cos(bax.tilt * np.pi / 180)
    ds = []
    for ax in bax.axs:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
        ax.tick_params(labelsize=20, width=2, length=6)
        second_axis(ax, "top", tick_spacing_x)
        if ax.get_subplotspec().is_last_col():
            second_axis(ax, "right", tick_spacing_y)
        bounds = ax.get_position().bounds
        if ax.get_subplotspec().is_first_row():
            ypos = bounds[1] + bounds[3]
            if not ax.get_subplotspec().is_last_col():
                xpos = bounds[0] + bounds[2]
                ds += bax.draw_diag(ax, xpos, xlen, ypos, ylen, **d_kwargs)
            if not ax.get_subplotspec().is_first_col():
                xpos = bounds[0]
                ds += bax.draw_diag(ax, xpos, xlen, ypos, ylen, **d_kwargs)


def plot_2(measure, sim_values, n, ylim, tick_spacing_y):
    """
    Generation and customization of figure 3 of "simulation_calculation_convex_hull".

    Parameters
    ----------
    measure : str
        One of "peri", "area", "surf_area", "volume".
    sim_values : numpy.ndarray
        Result of convex_hull.sim_{measure}_convex_hull_{dimension}d.
    n : int
        Amount of points each convex hull includes and consists of.
    ylim : float
        Highest y-value.
    tick_spacing_y : float
        The base of matplotlib.ticker.MultipleLocator of the y-axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure 3.
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis of figure 3.
    """
    rcParams["axes.linewidth"] = 2
    fig, ax = plt.subplots(figsize=(3, 2))
    x = np.linspace(0, 3, 100)
    if measure == "peri":
        ratio = sim_values[n - 3] / ch.calc_peri_convex_hull_2d_pt2(1, n)
        ax.set_xlabel(r"$P_{ch}/E(P_{ch})$", fontsize=21)
    elif measure == "area":
        ratio = sim_values[n - 3] / ch.calc_area_convex_hull_2d_pt2(1, n)
        ax.set_xlabel(r"$A_{ch}/E(A_{ch})$", fontsize=21)
    elif measure == "surf_area":
        ratio = sim_values[n - 4] / ch.calc_area_convex_hull_3d_pt2(1, 1, n)
        ax.set_xlabel(r"$S_{ch}/E(S_{ch})$", fontsize=21)
    elif measure == "volume":
        ratio = sim_values[n - 4] / ch.calc_volume_convex_hull_3d_pt2(1, 1, n)
        ax.set_xlabel(r"$V_{ch}/E(V_{ch})$", fontsize=21)
    else:
        ratio = None
    std = np.std(ratio)
    ax.hist(x=ratio, bins=20, range=(0, 3), density=True)
    ax.plot(x, norm.pdf(x=x, loc=1, scale=std), color="k", lw=2)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("PD", fontsize=21)
    ax.tick_params(labelsize=20, width=2, length=6)
    tick_spacing_x = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))
    second_axis(ax, "top", tick_spacing_x)
    second_axis(ax, "right", tick_spacing_y)

    return fig, ax
