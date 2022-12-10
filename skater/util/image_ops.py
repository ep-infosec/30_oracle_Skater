# -*- coding: UTF-8 -*-

import numpy as np
import skimage
import skimage.io
from scipy import ndimage
from skimage import util
from skimage.transform import rotate
from skimage import exposure

from .exceptions import MatplotlibUnavailableError
from .exceptions import DataSetError
from skater.util.logger import build_logger
from skater.util.logger import _INFO

__all__ = ['add_noise', 'image_transformation', 'flip_pixels', 'normalize', 'show_image', 'greater_than', 'less_than',
           'equal_to', 'greater_than_or_equal', 'less_than_equal', 'in_between', 'view_windows']


logger = build_logger(_INFO, __name__)


def load_image(path, img_height=None, img_width=None, crop_from_center=True, rgb2bgr=False, mode='constant',
               preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None):
    if not img_height or img_width is None:
        raise AssertionError("Dimensions for the image size is not specified as reqiured. Either image height or "
                             "image weight is missing ")

    # load image
    img = skimage.io.imread(path)

    # Crop image from the center
    short_edge = np.min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]

    in_img = crop_img if crop_from_center is True else img

    # If RGB --> BGR flag is enabled
    in_img = in_img[:, :, ::-1] if rgb2bgr else in_img
    # Re-size the image to required dimension
    resized_img = skimage.transform.resize(in_img, (img_height, img_width), mode=mode,
                                           preserve_range=preserve_range, anti_aliasing=anti_aliasing,
                                           anti_aliasing_sigma=anti_aliasing_sigma)
    return resized_img


def add_noise(image, noise_typ='gaussian', random_state=None):
    """ Helper function to add different forms of noise

    Parameters
    -----------
    image: numpy.ndarray
        input image
    noise_typ: string ( default 'gaussian' )
        - 'gaussian'  Gaussian-distributed additive noise.
        - 'localvar'  Gaussian-distributed additive noise, with specified
                      local variance at each point of `image`
        - 'poisson'   Poisson-distributed noise generated from the data.
        - 'salt'      Replaces random pixels with 1.
        - 'pepper'    Replaces random pixels with 0 (for unsigned images) or
                      -1 (for signed images).
        - 's&p'       Replaces random pixels with either 1 or `low_val`, where
                      `low_val` is 0 for unsigned images or -1 for signed
                      images.
        - 'speckle'   Multiplicative noise using out = image + n*image, where
                      n is uniform noise with specified mean & variance.
    random_state: int
        seed for setting repeatable state before generating noise

    Returns
    -------
    numpy.ndarray

    References
    ----------
    .. [1] http://scikit-image.org/docs/stable/api/skimage.util.html#random-noise
    """
    return skimage.util.random_noise(image, mode=noise_typ, seed=random_state)


def _rescale_intensity(X, q=(0.2, 99.8)):
    v_min, v_max = np.percentile(X, q)
    return exposure.rescale_intensity(X, in_range=(v_min, v_max))


def image_transformation(X, method_type='blur', **kwargs):
    # https://www.kaggle.com/tomahim/image-manipulation-augmentation-with-skimage
    q = kwargs['percentile'] if 'percentile' in kwargs else (0.2, 99.8)
    angle = kwargs['angle'] if 'angle' in kwargs else 60
    transformation_dict = {
        'blur': normalize(ndimage.uniform_filter(X)),
        'invert': normalize(util.invert(X)),
        'rotate': rotate(X, angle=angle),
        'rescale_intensity': _rescale_intensity(X, q=q),
        'gamma_correction': exposure.adjust_gamma(X, gamma=0.4, gain=0.9),
        'log_correction': exposure.adjust_log(X),
        'sigmoid_correction': exposure.adjust_sigmoid(X),
        'horizontal_flip': X[:, ::-1],
        'vertical_flip': X[::-1, :],
        'rgb2gray': skimage.color.rgb2gray(X)
    }
    return transformation_dict[method_type]


# Helper functions for filtering based on conditional type
greater_than = lambda X, value: np.where(X > value)
less_than = lambda X, value: np.where(X < value)
equal_to = lambda X, value: np.where(X == value)
greater_than_or_equal = lambda X, value: np.where(X >= value)
less_than_equal = lambda X, value: np.where(X < value)
in_between = lambda X, min_value, max_value: np.where((X >= min_value) & (X <= max_value))


def flip_pixels(X, num_of_pixel, filtered_pixel=None, replace_with=0, random_state=0):
    # make a deep copy of the original image
    import copy
    X_modified = copy.deepcopy(X)
    np.random.seed(random_state)
    try:
        if len(filtered_pixel) > 0 & isinstance(filtered_pixel, tuple):
            f_pixels = filtered_pixel
            logger.info("Number of pixels matching the condition : {}".format(len(f_pixels[0])))
            logger.info("Number of pixels specified to be replaced : {}".format(num_of_pixel))

            if len(f_pixels) == 3:
                # uniformly random
                h = np.random.choice(f_pixels[0], num_of_pixel, replace=False)
                w = np.random.choice(f_pixels[1], num_of_pixel, replace=False)
                c = np.random.choice(f_pixels[2], num_of_pixel, replace=False)

                # for the selected pixels, set the pixel intensity to 0
                for h_i, w_i, c_i in zip(h, w, c):
                    X_modified[h_i, w_i, c_i] = replace_with
            elif len(f_pixels) == 2:
                # uniformly random
                h = np.random.choice(f_pixels[0], num_of_pixel, replace=False)
                w = np.random.choice(f_pixels[1], num_of_pixel, replace=False)

                # for the selected pixels, set the pixel intensity to 0
                for h_i, w_i in zip(h, w):
                    X_modified[h_i, w_i] = replace_with
            else:
                logger.info("Ambiguity in the shape of the input image : {}".format(X.shape))
    except:
        logger.info("No matching pixel for the specified condition")
    return X_modified


def flip_orientation(X, axis=0):
    if not isinstance(X, np.ndarray):
        err_msg = "Input array should be numpy.ndarray"
        raise(DataSetError(err_msg))

    return np.flip(X, axis)


def normalize(X):
    """ Normalize image of the shape (H, W, D) in the range of 0 and 1 (Min-Max scaling)
    """
    if not isinstance(X, np.ndarray):
        err_msg = "Input array should be numpy.ndarray"
        raise(DataSetError(err_msg))

    return np.array((X - np.min(X)) / (np.max(X) - np.min(X)))


def standard_scaler(X):
    if not isinstance(X, np.ndarray):
        err_msg = "Input array should be numpy.ndarray"
        raise(DataSetError(err_msg))

    X = (X - X.mean()) / X.std()
    return X


def show_image(X, cmap=None, bins=None, intensity_plot=False, width=8, height=8, title='Original'):
    # TODO: Add ability to handle a pre-defined axes for plotting
    import copy
    font = {'family': 'avenir',
            'color': 'black',
            'weight': 'normal',
            'size': 14,
            }
    _X = copy.deepcopy(X)

    try:
        import matplotlib.pyplot as plt
        n_clmns = 2 if intensity_plot else 1
        fig, ax = plt.subplots(1, n_clmns, figsize=(width, height))
        if intensity_plot:
            ax[0].imshow(_X, cmap=cmap)
            ax[0].set_title(title)
            ax[1].hist(_X.ravel(), bins=bins, histtype='step')
            ax[1].set_xlabel('Pixel intensity', fontdict=font)
            fig.subplots_adjust(wspace=0.3)
        else:
            ax.imshow(_X, cmap=cmap)
            ax.set_title(title)
    except ImportError:
        raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on the system."))


padding_3d = lambda X, n_r, n_col, n_c: np.pad(X, ((0, n_r), (0, n_col), (0, n_c)), mode='constant', constant_values=0)
padding_2d = lambda X, n_r, n_col: np.pad(X, ((0, n_r), (0, n_col)), mode='constant', constant_values=0)


def view_windows(X, window_shape, steps):
    # Reference: https://codereview.stackexchange.com/questions/78156/moving-window-with-complete-boundary-in-python
    from skimage.util.shape import view_as_windows
    # number of rows to expand
    n_rows = np.mod(X.shape[0], steps)
    # number of columns to expand
    n_cols = np.mod(X.shape[1], steps)
    n_channel = np.mod(X.shape[2], steps) if len(window_shape) == 3 else None
    X_extended = padding_3d(X, n_rows, n_cols, n_channel) if n_channel is not None else padding_2d(X, n_rows, n_cols)
    rolling_view = view_as_windows(X_extended, window_shape, steps)
    return rolling_view
