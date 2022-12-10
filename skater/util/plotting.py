import numpy as np

from .exceptions import MatplotlibDisplayError, MatplotlibUnavailableError

COLORS = ['#328BD5', '#404B5A', '#3EB642', '#E04341', '#8665D0']


class LinearSegments(object):

    black_to_blue_dict = {
        'red': ((0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (1.0, 1.0, 0.0)),
        'green': ((0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}

    black_to_green_dict = {
        'red': ((0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0),
                  (1.0, 1.0, 0.0))}

    red_to_green_dict = {
        'red': ((0.0, 1.0, 1.0),
                (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0),
                  (1.0, 1.0, 0.0))}


class ColorMap(object):
    """
    Maps arrays to colors
    """
    def __init__(self):
        try:
            global colors
            global cm
            from matplotlib import colors, cm
        except ImportError:
            raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on your system."))
        except RuntimeError:
            raise (MatplotlibDisplayError("Matplotlib unable to open display"))

        self.red_to_green = colors.LinearSegmentedColormap('my_colormap', LinearSegments.red_to_green_dict, 100)
        self.black_to_blue = colors.LinearSegmentedColormap('my_colormap', LinearSegments.black_to_blue_dict, 100)
        self.black_to_green = colors.LinearSegmentedColormap('my_colormap', LinearSegments.black_to_green_dict, 100)


    def array_1d_to_color_scale(self, array_1d, colormap):
        mmin, mmax = min(array_1d), max(array_1d)
        norm = colors.Normalize(mmin, mmax)
        scalarMapx = cm.ScalarMappable(norm=norm, cmap=colormap)
        scalarMapx.set_array(array_1d)
        return scalarMapx.to_rgba(array_1d)


def coordinate_gradients_to_1d_colorscale(dx, dy,
                                          x_buffer_prop=.1, y_buffer_prop=.1,
                                          norm='separate'):
    """
    Map x and y gradients to single array of colors based on 2D color scale

    Parameters
    ----------
    dx: array type
        x component of gradient
    dy: array type
        y component of gradient
    x_buffer_prop:
    y_buffer_prop:
    norm: string
        whether to normalize colors based on differences in x and y scales.
        if separate, each component gets its own scaling (default)
        if global, will scale based on global extrema

    Returns
    ----------
    color_array, xmin, xmax, ymin, ymax
    """
    colormap = ColorMap()
    xmin, xmax, xbuffer = build_buffer(dx.min(), dx.max())
    ymin, ymax, ybuffer = build_buffer(dy.min(), dy.max())
    global_min = min(xmin, ymin)
    global_max = max(xmax, ymax)
    if norm == 'separate':
        normx = colors.Normalize(xmin, xmax)
        normy = colors.Normalize(ymin, ymax)
    elif norm == 'shared':
        normx = colors.Normalize(global_min, global_max)
        normy = colors.Normalize(global_min, global_max)
    else:
        raise KeyError("keyword norm must be in ('separate', 'shared')")

    scalarMapx = cm.ScalarMappable(norm=normx, cmap=colormap.black_to_blue)
    scalarMapy = cm.ScalarMappable(norm=normy, cmap=colormap.black_to_green)

    scalarMapx.set_array(dx)
    scalarMapy.set_array(dy)

    colorx = scalarMapx.to_rgba(dx)
    colory = scalarMapy.to_rgba(dy)

    color = np.array(colorx) + np.array(colory)
    color[:, :, 3] = 1.
    xmin, xmax, ymin, ymax = xmin + xbuffer, xmax - xbuffer, ymin + ybuffer, ymax - ybuffer
    return color, xmin, xmax, ymin, ymax


def plot_2d_color_scale(x1_min, x1_max, x2_min, x2_max, plot_point=None,
                        resolution=10, ax=None, x_buffer_prop=.1, y_buffer_prop=.1):
    """
    Return a generic plot of a 2D color scale

    Parameters
    ----------
    x1_min: numeric
        how high x1 should extend the color scale
    x1_max:  numeric
        how high 2 should extend the color scale
    x2_min:  numeric
        how low x2 should extend the color scale
    x2_max: numeric
        how high x2 should extend the color scale
    resolution: int
        how fine grain to make the color scale
    ax: matplotlib.axes._subplots.AxesSubplot
        matplotlib axis to plot on. if none will generate a new one

    Returns
    ----------
    matplotlib.axes._subplots.AxesSubplot
    """

    try:
        global pyplot
        global patches
        from matplotlib import pyplot, patches
    except ImportError:
        raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on your system."))
    except RuntimeError:
        raise (MatplotlibDisplayError("Matplotlib unable to open display"))

    # xmin, xmax = min(0, np.percentile(x, 3)), max(np.percentile(x, 97), 0)

    if ax is None:
        f, ax = pyplot.subplots(1)

    x1_min, x1_max, x1_buffer = build_buffer(x1_min, x1_max, buffer_prop=x_buffer_prop)
    x2_min, x2_max, x2_buffer = build_buffer(x2_min, x2_max, buffer_prop=y_buffer_prop)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    x1 = np.linspace(x1_min, x1_max, resolution + 1)
    x2 = np.linspace(x2_min, x2_max, resolution + 1)
    x1_diff = x1[1] - x1[0]
    x2_diff = x2[1] - x2[0]
    x1, x2 = np.meshgrid(x1, x2)
    colors_for_scale, a, b, c, d = coordinate_gradients_to_1d_colorscale(x1,
                                                                         x2,
                                                                         x_buffer_prop=0,
                                                                         y_buffer_prop=0)

    if ax is None:
        f, ax = pyplot.subplots(1)
    for i in range(resolution):
        for j in range(resolution):
            xy = (x1[i, j], x2[i, j])
            color = colors_for_scale[i, j]
            rect = patches.Rectangle(
                xy, x1_diff, x2_diff,
                alpha=.8, facecolor=color, zorder=1
            )
            ax.add_patch(rect)
    if plot_point:
        ax.scatter([plot_point[0]], [plot_point[1]], s=75.,
                   color='yellow', alpha=1, zorder=2)
    ax.grid()
    return ax


def build_buffer(xmin, xmax, buffer_prop=.1):
    if xmin != xmax:
        buffer = (xmax - xmin) * buffer_prop / 2.
    # all values are the same
    else:
        buffer = 0.01
    xmin, xmax = xmin - buffer, xmax + buffer
    return xmin, xmax, buffer


def tick_formatter(powerlimits=None):
    try:
        from matplotlib.ticker import ScalarFormatter
    except ImportError:
        raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on your system."))
    except RuntimeError:
        raise (MatplotlibDisplayError("Matplotlib unable to open display"))
    if powerlimits is None:
        powerlimits = (3, 3)
    formatter = ScalarFormatter()
    formatter.set_powerlimits(powerlimits)
    return formatter


def plot_from_df(data_df, column_name_to_sort, x_label, y_label, ascending=True, display_size=(8, 8)):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on your system."))

    plt.style.use('ggplot')
    f, ax = plt.subplots(1, figsize=display_size)
    data_df.sort_values(column_name_to_sort, ascending=ascending).plot(kind='barh', ax=ax,
                                                                       color='blue', title='Feature ranking')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_yticklabels(data_df.features)
    plt.show()
