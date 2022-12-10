from __future__ import division
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from skater.util.logger import build_logger
from skater.util.logger import _INFO

logger = build_logger(_INFO, __name__)


def flatten(array):
    return [item for sublist in array for item in sublist]


def add_column_numpy_array(array, new_col):
    placeholder = np.ones(array.shape[0])[:, np.newaxis]
    result = np.hstack((array, placeholder))

    if isinstance(new_col, np.ndarray):
        assert array.shape[0] == new_col.shape[0], "input array row counts \
                                                    must be the same. \
                                                    Expected: {0}\
                                                    Actual: {1}".format(array.shape[0],
                                                                        new_col.shape[0])
        assert len(new_col.shape) <= 2, "new column must be 1D or 2D"

        if len(new_col.shape) == 1:
            new_col = new_col[:, np.newaxis]
        return np.hstack((array, new_col))
    elif isinstance(new_col, list):
        assert len(new_col) == array.shape[0], "input array row counts \
                                                    must be the same. \
                                                    Expected: {0}\
                                                    Actual: {1}".format(len(array),
                                                                        len(new_col))
        new_col = np.array(new_col)
        assert len(new_col.shape) == 1, "list elements cannot be iterable"
        new_col = new_col[:, np.newaxis]
        return np.hstack((array, new_col))
    else:
        placeholder = np.ones(array.shape[0])[:, np.newaxis]
        result = np.hstack((array, placeholder))
        result[:, -1] = new_col
        return result


def allocate_samples_to_bins(n_samples, ideal_bin_count=100):
    """goal is as best as possible pick a number of bins
    and per bin samples to a achieve a given number
    of samples.

    Parameters
    ----------

    Returns
    ----------
    number of bins, list of samples per bin
    """

    if n_samples <= ideal_bin_count:
        n_bins = n_samples
        samples_per_bin = [1 for _ in range(n_bins)]
    else:
        n_bins = ideal_bin_count
        remainer = n_samples % ideal_bin_count

        samples_per_bin = np.array([(n_samples - remainer) / ideal_bin_count for _ in range(n_bins)])
        if remainer != 0:
            additional_samples_per_bin = distribute_samples(remainer, n_bins)
            samples_per_bin = samples_per_bin + additional_samples_per_bin
    return n_bins, np.array(samples_per_bin).astype(int)


def distribute_samples(n_samples, n_bins):
    assert n_samples < n_bins, "number of samples should be \
                                less than number of bins"
    space_size = n_bins / n_samples

    samples_per_bin = np.zeros(n_bins).tolist()

    index_counter = 0
    for sample in range(n_samples):
        index = int(index_counter)
        samples_per_bin[index] += 1
        index_counter += space_size
    return np.array(samples_per_bin).astype(int)


def divide_zerosafe(a, b):
    """ diving by zero returns 0 """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


# Lambda for converting data-frame to a dictionary
convert_dataframe_to_dict = lambda key_column_name, value_column_name, df: \
    df.set_index(key_column_name).to_dict()[value_column_name]


def json_validator(json_object):
    """ json validator
    """
    # Reference: https://stackoverflow.com/questions/5508509/how-do-i-check-if-a-string-is-valid-json-in-python
    import json
    try:
        json.loads(json_object)
    except ValueError:
        return False
    return True


def _render_html(file_name, width=None, height=None):
    width, height
    from IPython.core.display import HTML
    return HTML(file_name)


def _render_image(file_name, width=600, height=300):
    from IPython.display import Image
    return Image(file_name, width=width, height=height)


def _render_pdf(file_name, width=600, height=300):
    from IPython.display import IFrame
    IFrame(file_name, width=width, height=height)


def show_in_notebook(file_name_with_type='rendered.html', width=600, height=300, mode=None):
    """ Display generated artifacts(e.g. .png, .html, .jpeg/.jpg) in interactive Jupyter style Notebook

    Parameters
    -----------
    file_name_with_type: str
        specify the name of the file to display
    width: int
        width in pixels to constrain the image
    height: int
        height in pixels to constrain the image
    """
    from IPython.core.display import display, HTML
    if mode is not 'interactive':
        file_type = file_name_with_type.split('/')[-1].split('.')[-1]
        choice_dict = {
            'html': _render_html,
            'png': _render_image,
            'jpeg': _render_image,
            'jpg': _render_image,
            'svg': _render_image,
            'pdf': _render_pdf
        }
        select_type = lambda choice_type: choice_dict[file_type]
        logger.info("File Name: {}".format(file_name_with_type))
        return display(select_type(file_type)(file_name_with_type, width, height))
    else:
        # For now using iframe for some interactive plotting. This should be replaced with a better plotting interface
        iframe_style = '<div style="-webkit-overflow-scrolling:touch; overflow-x:hidden; ' \
                       'overflow-y:auto; width:{}px; height:{}px; margin: -1.2em; ' \
                       '-webkit-transform: scale(0.9) -moz-transform-scale(0.5)"> ' \
                       '<iframe src={} style="width:100%; height:100%; frameborder:1px;">' \
                       '</iframe>' \
                       '</div>'.format(width, height, file_name_with_type)
        return HTML(iframe_style)


class MultiColumnLabelBinarizer(LabelBinarizer):
    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output
        self.binarizers = []


    def fit(self, X):
        for x in X.T:
            binarizer = LabelBinarizer()
            binarizer.fit(x)
            self.binarizers.append(binarizer)


    def transform(self, X):
        results = []
        for i, x in enumerate(X.T):
            results.append(self.binarizers[i].transform(x))
        return np.concatenate(results, axis=1)


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


    def inverse_transform(self, X):
        results = []
        column_counter = 0

        for i, binarizer in enumerate(self.binarizers):
            n_cols = binarizer.classes_.shape[0]
            x_subset = X[:, column_counter:column_counter + n_cols]
            inv = binarizer.inverse_transform(x_subset)
            if len(inv.shape) == 1:
                inv = inv[:, np.newaxis]
            results.append(inv)
            column_counter += n_cols
        return np.concatenate(results, axis=1)
