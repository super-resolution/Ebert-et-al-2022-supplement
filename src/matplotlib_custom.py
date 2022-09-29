import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np


def second_axis_log(ax, location, tick_spacing):
    """
    Generates an additional logarithmic axis at location of subplot ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot axis.
    location : str
        One of "top", "right".
    tick_spacing : float
        The base of matplotlib.ticker.LogLocator.

    Returns
    -------
    sec_ax : matplotlib.axes._secondary_axes.SecondaryAxis
        The additional axis.
    """
    if location == "top":
        sec_ax = ax.secondary_xaxis(location)
        sec_ax.set_xscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
        sec_ax.tick_params(width=2, axis="x", direction="in", labeltop=False, length=6)
        sec_ax.tick_params(which="minor", axis="x", width=2, direction="in", length=4)
        if tick_spacing:
            sec_ax.xaxis.set_major_locator(ticker.LogLocator(base=tick_spacing))
    elif location == "right":
        sec_ax = ax.secondary_yaxis(location)
        sec_ax.set_yscale("log", subs=[10])
        sec_ax.tick_params(width=2, axis="y", direction="in", labelright=False, length=6)
        sec_ax.tick_params(which="minor", axis="y", width=2, direction="in", length=4, labelright=False)
        if tick_spacing:
            sec_ax.yaxis.set_major_locator(ticker.LogLocator(base=tick_spacing))
    else:
        sec_ax = None

    return sec_ax


def second_axis(ax, location, tick_spacing=None, num_ticks=None):
    """
    Generates an additional axis at location of subplot ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot axis.
    location : str
        One of "top", "right".
    tick_spacing : float
        The base of matplotlib.ticker.MultipleLocator.
    num_ticks : int
        The number of ticks evenly placed within the axis range.

    Returns
    -------
    sec_ax : matplotlib.axes._secondary_axes.SecondaryAxis
        The additional axis.
    """
    if location == "top":
        sec_ax = ax.secondary_xaxis(location)
        sec_ax.tick_params(width=2, axis="x", direction="in", labeltop=False, length=6)
        if tick_spacing:
            sec_ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        elif num_ticks:
            sec_ax.xaxis.set_major_locator(ticker.LinearLocator(num_ticks))
    elif location == "right":
        sec_ax = ax.secondary_yaxis(location)
        sec_ax.tick_params(width=2, axis="y", direction="in", labelright=False, length=6)
        if tick_spacing:
            sec_ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        elif num_ticks:
            sec_ax.yaxis.set_major_locator(ticker.LinearLocator(num_ticks))
    else:
        sec_ax = None

    return sec_ax


def pie_plot(sizes, labels, colors, labeldistance, radius, textprops, hatches, rounded_texts, rt_size, rt_h_space,
             rt_v_space, wedgeallignments, stretch, startangle):
    """
    Generation and customization of pie plot. Used to generate rounded texts aligned to the circumference of the pie
    plot.

    Parameters
    ----------
    sizes : numpy.ndarray
        Contains the wedge sizes in percent.
    labels : list
        Contains the labels of the individual wedges.
    colors : list
        Contains the colors of the individual wedges.
    labeldistance : float
        Relative distance of label to radius.
    radius : float
        Radius of the circular pie plot.
    textprops : dict
        Contains arguments of text objects to apply to labels.
    hatches : list
        Contains the hatches of the individual wedges.
    rounded_texts : list
        Contains text for the individual wedges, displayed outside of the wedges in rounded mode.
    rt_size : float
        The text size of rounded_texts.
    rt_h_space : float
        The horizontal space between characters of a rounded text.
    rt_v_space : float
        The vertical space between rounded text and pie chart. Relation to radius.
    wedgeallignments : list
        Contains the relative starting point of a rounded text.
    stretch : float
        A factor to increase the axis size to create space for rounded text display.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the pie plot.
    ax : matplotlib.axes._subplots.AxesSubplot
        Axis of the pie plot.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    pie = ax.pie(x=sizes, labels=labels, colors=colors, labeldistance=labeldistance, radius=radius,
                 startangle=startangle, counterclock=False, textprops=textprops)

    xlim = ax.get_xlim()
    w = xlim[1] - xlim[0]
    ax.set_xlim([xlim[0] - stretch * w, xlim[1] + stretch * w])
    ylim = ax.get_ylim()
    h = ylim[1] - ylim[0]
    ax.set_ylim([ylim[0] - stretch * h, ylim[1] + stretch * h])

    for wedge, hatch in zip(pie[0], hatches):
        wedge.set_hatch(hatch)
        wedge.set_linewidth(1)
        wedge.set_edgecolor("k")

    for i, texts in enumerate(pie[1]):
        texts.set(backgroundcolor=colors[i])

    radians = sizes / 100 * 2 * np.pi
    radians = np.insert(radians, 0, -np.deg2rad(startangle) + np.pi)
    radians_cumsum = np.cumsum(radians)

    for j, (text, wedgeallignment) in enumerate(zip(rounded_texts, wedgeallignments)):
        if text:
            x = rt_v_space * radius * (-np.cos(np.linspace(radians_cumsum[j], radians_cumsum[j] + 2*np.pi, 360)))
            y = rt_v_space * radius * (np.sin(np.linspace(radians_cumsum[j], radians_cumsum[j] + 2*np.pi, 360)))
            starting_angle = np.rad2deg(radians_cumsum[j])
            degs = np.linspace(starting_angle, starting_angle+360, 360)

            for i, character in enumerate(text):
                i = i * rt_h_space + wedgeallignment
                ax.text(x[i], y[i], character, rotation=-degs[i] + 90, rotation_mode="anchor", va="bottom", ha="center",
                        size=rt_size)

    return fig, ax
