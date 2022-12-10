import copy
import random
import numpy as np
import pandas as pd


from skater.util import exceptions

try:
    import plotly.offline as py
    py.init_notebook_mode(connected=True)
    import plotly.graph_objs as go
    from plotly import tools
except ImportError:
    raise exceptions.PlotlyUnavailableError("plotly is required but unavailable on your system.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib import colors as mcolors
except ImportError:
    raise exceptions.MatplotlibUnavailableError("Matplotlib is required but unavailable on your system.")

_enable_axis = lambda ax, flag: ax.axis("on") if flag is True else ax.axis("off")


def _create_meshgrid(xx, yy, plot_step=0.02):
    xmin, xmax = xx.min() - 0.5, xx.max() + 0.5
    ymin, ymax = yy.min() - 0.5, yy.max() + 0.5
    xx, yy = np.meshgrid(np.arange(xmin, xmax, plot_step),
                         np.arange(ymin, ymax, plot_step))
    x_ = pd.DataFrame({'F1': xx.ravel(), 'F2': yy.ravel()})
    return x_, xx, yy


def _generate_contours(est, X_, xx, yy, color_map, ax, **params):
    Z = est.predict(X_).reshape(xx.shape)
    cf = ax.contourf(xx, yy, Z, alpha=0.8, cmap=color_map, **params)
    return cf


# Reference: https://plot.ly/scikit-learn/plot-voting-decision-regions/
def interactive_plot(est, X0, X1, Y, x_label="X1", y_label="X2", title="decision_boundary",
                     file_name='decision_iplot', height=10, width=10):
    figure = tools.make_subplots(rows=1, cols=1, print_grid=False)

    X_, xx, yy = _create_meshgrid(X0, X1)
    Z = est.predict(X_).reshape(xx.shape)

    # generate the contour
    trace1 = go.Contour(x=xx[0], y=yy[:, 1], z=Z, colorscale='Viridis', opacity=0.2, showscale=False)

    # Scatter plot is generated using the original specified data points
    trace2 = go.Scatter(x=X0, y=X1, showlegend=False, mode='markers',
                        marker=dict(color=Y, line=dict(color='black', width=1),
                                    colorscale='Viridis', showscale=True))

    figure.append_trace(trace1, 1, 1)
    figure.append_trace(trace2, 1, 1)

    layout = go.Layout(
        xaxis=dict(autorange=True,
                   showgrid=False,
                   zeroline=False,
                   showline=True,
                   ticks='',
                   showticklabels=True,
                   title=x_label),
        yaxis=dict(autorange=True,
                   showgrid=False,
                   zeroline=False,
                   showline=True,
                   ticks='',
                   showticklabels=True,
                   title=y_label),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        width=width,
        height=height,
        title=title
    )

    figure.update(layout=layout)
    py.iplot(figure, filename=file_name)
    return py, figure


def plot_decision_boundary(est, X0, X1, Y, mode='static', width=12, height=10,
                           retrain=True, title='decision_boundary',
                           x0_label='X1', x1_label='X2', feature_names=None,
                           static_color_map=None, enable_axis=False,
                           file_name='decision_plot', random_state=0, **params):
    f_n = ['F1', 'F2'] if feature_names is None else feature_names
    x0_label = feature_names[0] if x0_label is None else x0_label
    x1_label = feature_names[1] if x1_label is None else x1_label
    X = pd.concat([X0, X1], keys=f_n, axis=1)

    est = copy.deepcopy(est)
    if retrain:
        est.fit(X, Y)

    if mode == 'static':
        colors = list(mcolors.CSS4_COLORS.keys())
        n_classes = len(np.unique(Y))
        random.seed(random_state)
        random.shuffle(colors)
        color_list = colors[:n_classes]
        cm = ListedColormap(color_list) if static_color_map is None \
            else ListedColormap(static_color_map)

        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        X_grid, xx, yy = _create_meshgrid(X0, X1)
        cs = _generate_contours(est, X_grid, xx, yy, cm, ax, **params)
        # set other properties of the plot
        ax.scatter(X0, X1, c=Y, cmap=cm, alpha=0.6, linewidths=0.9, edgecolors='white')
        ax.set_xlim(X_grid.iloc[:, 0].min(), X_grid.iloc[:, 0].max())
        ax.set_ylim(X_grid.iloc[:, 1].min(), X_grid.iloc[:, 1].max())
        ax.set_xlabel(x0_label)
        ax.set_ylabel(x1_label)
        ax.set_title(title)
        _enable_axis(ax, enable_axis)
        fig.colorbar(cs, ax=ax, shrink=0.9)
        fig.savefig('{}.png'.format(file_name))
        return fig, ax
    else:  # interactive mode
        fig = plt.gcf()
        # using matplotlib dpi to convert from inches to pixels
        dpi = fig.get_dpi()
        return interactive_plot(est, X0, X1, Y, x0_label, x1_label, title,
                                file_name, height * dpi, width * dpi)
