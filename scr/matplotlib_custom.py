import matplotlib.ticker as ticker


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


def second_axis(ax, location, tick_spacing):
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
    elif location == "right":
        sec_ax = ax.secondary_yaxis(location)
        sec_ax.tick_params(width=2, axis="y", direction="in", labelright=False, length=6)
        if tick_spacing:
            sec_ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    else:
        sec_ax = None

    return sec_ax
