# -*- coding: UTF-8 -*-

from skimage.filters import roberts, sobel
import numpy as np
from skater.util.exceptions import MatplotlibUnavailableError
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on your system."))

from skater.util.image_ops import normalize

# helper function to enable or disable matplotlib access
_enable_axis = lambda ax, flag: ax.axis("off") if flag is True else ax.axis("on")


def visualize(relevance_score, original_input_img=None, edge_detector_type='sobel', cmap='bwr', axis=plt,
              percentile=100, alpha_edges=0.8, alpha_bgcolor=1, disable_axis=True):

    dx, dy = 0.01, 0.01
    xx = np.arange(0.0, relevance_score.shape[1], dx)
    yy = np.arange(0.0, relevance_score.shape[0], dy)

    x_min, x_max, y_min, y_max = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = x_min, x_max, y_min, y_max
    xi_cmap = plt.cm.gray
    xi_cmap.set_bad(alpha=0)

    edges = _edge_detection(original_input_img, edge_detector_type) if original_input_img is not None else None

    # draw the edges of the image before overlaying rest of the image
    if edges is not None:
        axis.imshow(edges, extent=extent, interpolation='nearest', cmap=xi_cmap, alpha=alpha_edges)

    abs_max = np.percentile(np.abs(relevance_score), percentile)
    abs_min = abs_max

    relevance_score = relevance_score[:, :, 0] if len(relevance_score.shape) == 3 else relevance_score
    # Plot the image with relevance scores
    axis.imshow(relevance_score, extent=extent, interpolation='nearest', cmap=cmap,
                vmin=-abs_min, vmax=abs_max, alpha=alpha_bgcolor)
    _enable_axis(axis, disable_axis)
    return axis


def _edge_detection(original_input_img=None, edge_detector_alg='sobel'):
    # Normalize the input image to (0,1)
    xi = normalize(original_input_img)
    xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
    # Applying edge detection ( Roberts or Sobel edge detection )
    # Reference: http://scikit-image.org/docs/0.11.x/auto_examples/plot_edge_filter.html
    edge_detector = {'robert': roberts, 'sobel': sobel}
    return edge_detector[edge_detector_alg](xi_greyscale)
