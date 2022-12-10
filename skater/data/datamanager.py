from __future__ import division
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import ast
from functools import partial

from ..util.logger import build_logger
from ..util import exceptions
from ..util.static_types import StaticTypes
from ..util.dataops import add_column_numpy_array, allocate_samples_to_bins, flatten

__all__ = ['DataManager']


class DataManager(object):
    """Module for passing around data to interpretation objects
    """

    # Todo: we can probably remove some of the keys from data_info, and have properties
    # executed as pure functions for easy to access metadata, such as n_rows, etc

    _n_rows = 'n_rows'
    _dim = 'dim'
    _feature_info = 'feature_info'
    _dtypes = 'dtypes'

    __attribute_keys__ = [_n_rows, _dim, _feature_info, _dtypes]
    __datatypes__ = (pd.DataFrame, pd.Series, np.ndarray)

    def _check_X(self, X):
        if not isinstance(X, self.__datatypes__):
            err_msg = 'Invalid Data: expected data to be a numpy array or pandas dataframe but got ' \
                      '{}'.format(type(X))
            raise(exceptions.DataSetError(err_msg))
        ndim = len(X.shape)
        self.logger.debug("__init__ data.shape: {}".format(X.shape))

        if ndim == 1:
            X = X[:, np.newaxis]

        elif ndim >= 3:
            err_msg = "Invalid Data: expected data to be 1 or 2 dimensions, " \
                      "Data.shape: {}".format(ndim)
            raise(exceptions.DataSetError(err_msg))
        return X


    def _check_y(self, y, X):
        """
        convert y to ndarray

        If y is a dataframe:
            return df.values as ndarray
        if y is a series
            return series.values as ndarray
        if y is ndarray:
            return self
        if y is a list:
            return as ndarray
        :param y:
        :param X:
        :return:
        """

        if y is None:
            return None

        assert len(X) == len(y), \
            "len(X) = {0} does not equal len(y) = {1}".format(len(X), len(y))

        if isinstance(y, (pd.DataFrame, pd.Series)):
            return y.values
        elif isinstance(y, np.ndarray):
            return y
        elif isinstance(y, list):
            return np.array(y)
        else:
            raise ValueError("Unrecognized type for y: {}".format(type(y)))


    def __init__(self, X, y=None, feature_names=None, index=None, log_level=30):
        """
        The abstraction around using, accessing, sampling data for interpretation purposes.
        Used by interpretation objects to grab data, collect samples, and handle
        feature names and row indices.

        Parameters
        ----------
            X: 1D/2D numpy array, or pandas DataFrame
                raw data
            y: 1D/2D numpy array, or pandas DataFrame
                ground truth labels for X
            feature_names: iterable of feature names
                Optional keyword containing names of features.
            index: iterable of row names
                Optional keyword containing names of indexes (rows).

        """

        # create logger
        self._log_level = log_level
        self.logger = build_logger(log_level, __name__)

        self.X = self._check_X(X)
        self.y = self._check_y(y, self.X)
        self.data_type = type(self.X)
        self.metastore = None

        self.logger.debug("after transform X.shape: {}".format(self.X.shape))

        if isinstance(self.X, pd.DataFrame):
            if feature_names is None:
                feature_names = self.X.columns.values
            if index is None:
                index = range(self.n_rows)
            self.X.index = index

        elif isinstance(self.X, np.ndarray):
            if feature_names is None:
                feature_names = range(self.X.shape[1])
            if index is None:
                index = range(self.n_rows)

        else:
            raise(ValueError("Invalid: currently we only support {}"
                             "If you would like support for additional data structures let us "
                             "know!".format(self.__datatypes__)))

        self.feature_ids = list(feature_names)
        self.index = list(index)
        self.data_info = {attr: None for attr in self.__attribute_keys__}


    def generate_grid(self, feature_ids, grid_resolution=100, grid_range=(.05, .95)):
        """
        Generates a grid of values on which to compute pdp. For each feature xi, for value
        yj of xi, we will fix xi = yj for every observation in X.

        Parameters
        ----------
            feature_ids(list):
                Feature names for which we'll generate a grid. Must be contained
                by self.feature_ids

            grid_resolution(int):
                The number of unique values to choose for each feature.

            grid_range(tuple):
                The percentile bounds of the grid. For instance, (.05, .95) corresponds to
                the 5th and 95th percentiles, respectively.

        Returns
        -------
        grid(numpy.ndarray): 	There are as many rows as there are feature_ids
                                There are as many columns as specified by grid_resolution
        """

        if not all(i >= 0 and i <= 1 for i in grid_range):
            err_msg = "Grid range values must be between 0 and 1 but got:" \
                      "{}".format(grid_range)
            raise(exceptions.MalformedGridRangeError(err_msg))

        if not isinstance(grid_resolution, int) and grid_resolution > 0:
            err_msg = "Grid resolution {} is not a positive integer".format(grid_resolution)
            raise(exceptions.MalformedGridRangeError(err_msg))

        if not all(feature_id in self.feature_ids for feature_id in feature_ids):
            missing_features = []
            for feature_id in feature_ids:
                if feature_id not in self.feature_ids:
                    missing_features.append(feature_id)
            err_msg = "Feature ids {} not found in DataManager.feature_ids".format(missing_features)
            raise(KeyError(err_msg))

        grid_range = [x * 100 for x in grid_range]
        bins = np.linspace(*grid_range, num=grid_resolution).tolist()
        grid = []
        for feature_id in feature_ids:
            data = self[feature_id]
            info = self.feature_info[feature_id]
            # if a feature is categorical (non numeric) or
            # has a small number of unique values, we'll just
            # supply unique values for the grid
            if info['unique'] < grid_resolution or info['numeric'] is False:
                vals = np.unique(data)
            else:
                vals = np.unique(np.percentile(data, bins))
            grid.append(vals)
        grid = np.array(grid)
        grid_shape = [(1, i) for i in [row.shape[0] for row in grid]]
        self.logger.info('Generated grid of shape {}'.format(grid_shape))
        return grid


    def sync_metadata(self):
        self.data_info[self._n_rows] = self.n_rows
        self.data_info[self._dim] = self.dim
        self.data_info[self._dtypes] = self.dtypes
        self.data_info[self._feature_info] = self._calculate_feature_info()


    def _calculate_n_rows(self):
        return self.X.shape[0]


    def _calculate_dim(self):
        return self.X.shape[1]

    @property
    def values(self):
        if self.data_type == pd.DataFrame:
            result = self.X.values
        else:
            result = self.X
        return result


    @property
    def dtypes(self):
        return pd.DataFrame(self.X, columns=self.feature_ids, index=self.index).dtypes


    @property
    def shape(self):
        return self.X.shape


    @property
    def n_rows(self):
        return self.shape[0]


    @property
    def dim(self):
        return self.shape[1]


    def _calculate_feature_info(self):
        feature_info = {}
        for feature in self.feature_ids:
            x = self[feature]
            samples = self.generate_column_sample(feature, n_samples=10)
            samples_are_numeric = map(StaticTypes.data_types.is_numeric, np.array(samples))
            is_numeric = all(samples_are_numeric)
            feature_info[feature] = {
                'type': self.dtypes.loc[feature],
                'unique': len(np.unique(x)),
                'numeric': is_numeric
            }
        return feature_info

    @property
    def feature_info(self):
        if self.data_info[self._feature_info] is None:
            self.data_info[self._feature_info] = self._calculate_feature_info()
        return self.data_info[self._feature_info]


    def _build_metastore(self):

        medians = np.median(self.X, axis=0).reshape(1, self.dim)

        # how far each data point is from the global median
        dists = cosine_distances(self.X, Y=medians).reshape(-1)

        sorted_index = [self.index[i] for i in dists.argsort()]

        return {'sorted_index': sorted_index}

    def __repr__(self):
        return self.X.__repr__()

    def __iter__(self):
        for i in self.feature_ids:
            yield i


    def __setitem__(self, key, newval):
        if issubclass(self.data_type, pd.DataFrame) or issubclass(self.data_type, pd.Series):
            self.__setcolumn_pandas__(key, newval)
        elif issubclass(self.data_type, np.ndarray):
            self.__setcolumn_ndarray__(key, newval)
        else:
            raise ValueError("Can't set item for data of type {}".format(self.data_type))
        self.sync_metadata()


    def __setcolumn_pandas__(self, i, newval):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        self.X[i] = newval


    def __setcolumn_ndarray__(self, i, newval):
        """if you passed in a pandas dataframe, it has columns which are strings."""

        if i in self.feature_ids:
            idx = self.feature_ids.index(i)
            self.X[:, idx] = newval
        else:
            self.X = add_column_numpy_array(self.X, newval)
            self.feature_ids.append(i)


    def __getitem__(self, key):
        if issubclass(self.data_type, pd.DataFrame) or issubclass(self.data_type, pd.Series):
            return self.__getitem_pandas__(key)
        elif issubclass(self.data_type, np.ndarray):
            return self.__getitem_ndarray__(key)
        else:
            raise ValueError("Can't get item for data of type {}".format(self.data_type))


    def __getitem_pandas__(self, i):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        return self.X[i]


    def __getitem_ndarray__(self, i):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        if StaticTypes.data_types.return_data_type(i) == StaticTypes.output_types.iterable:
            idx = [self.feature_ids.index(j) for j in i]
            return self.X[:, idx]
        elif StaticTypes.data_types.is_string(i) or StaticTypes.data_types.is_numeric(i):
            idx = self.feature_ids.index(i)
            return self.X[:, idx]
        else:
            raise(ValueError("Unrecongized index type: {}. This should not happen".format(type(i))))


    def __getrows__(self, idx):
        if self.data_type == pd.DataFrame:
            return self.__getrows_pandas__(idx)
        elif self.data_type == np.ndarray:
            return self.__getrows_ndarray__(idx)
        else:
            raise ValueError("Can't get rows for data of type {}".format(self.data_type))


    def __getrows_pandas__(self, idx):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        if StaticTypes.data_types.return_data_type(idx) == StaticTypes.output_types.iterable:
            i = [self.index.index(i) for i in idx]
        else:
            i = [self.index[idx]]
        return self.X.iloc[i]


    def __getrows_ndarray__(self, idx):
        """if you passed in a pandas dataframe, it has columns which are strings."""
        i = [self.index.index(i) for i in idx]
        return self.X[i]


    def generate_sample(self, sample=True, include_y=False, strategy='random-choice', n_samples=1000,
                        replace=True, bin_count=50):
        """ Method for generating data from the dataset.

        Parameters
        -----------
            sample : boolean
                If False, we'll take the full dataset, otherwise we'll sample.

            include_y: boolean (default=False)

            strategy: string (default='random-choice')
                Supported strategy types 'random-choice', 'uniform-from-percentile', 'uniform-over-similarity-ranks'

            n_samples : int (default=1000)
                Specifies the number of samples to return. Only implemented if strategy is "random-choice".

            replace : boolean (default=True)
                Bool for sampling with or without replacement

            bin_count : int
                If strategy is "uniform-over-similarity-ranks", then this is the number
                of samples to take from each discrete rank.
        """

        __strategy_types__ = ['random-choice', 'uniform-from-percentile', 'uniform-over-similarity-ranks']

        bin_count, samples_per_bin = allocate_samples_to_bins(n_samples, ideal_bin_count=bin_count)
        arg_dict = {
            'sample': sample,
            'strategy': strategy,
            'n_samples': n_samples,
            'replace': replace,
            'samples_per_bin': samples_per_bin,
            'bin_count': bin_count
        }
        self.logger.debug("Generating sample with args:\n {}".format(arg_dict))

        if not sample:
            idx = self.index

        if strategy == 'random-choice':
            idx = np.random.choice(self.index, size=n_samples, replace=replace)

        elif strategy == 'uniform-from-percentile':
            raise(NotImplementedError("We havent coded this yet."))

        elif strategy == 'uniform-over-similarity-ranks':
            sorted_index = self._build_metastore()['sorted_index']
            range_of_indices = list(range(len(sorted_index)))

            def aggregator(samples_per_bin, list_of_indicies):
                n = samples_per_bin[aggregator.count]
                result = str(np.random.choice(list_of_indicies, size=n).tolist())
                aggregator.count += 1
                return result

            aggregator.count = 0
            agg = partial(aggregator, samples_per_bin)

            cuts = pd.qcut(range_of_indices, [i / bin_count for i in range(bin_count + 1)])
            cuts = pd.Series(cuts).reset_index()
            indices = cuts.groupby(0)['index'].aggregate(agg).apply(lambda x: ast.literal_eval(x)).values
            indices = flatten(indices)
            idx = [self.index[i] for i in indices]
        else:
            raise ValueError("Strategy {0} not recognized, currently supported strategies: {1}".format(
                strategy,
                __strategy_types__
            ))
        if include_y:
            return self.__getrows__(idx), self._labels_by_index(idx)
        else:
            return self.__getrows__(idx)


    def generate_column_sample(self, feature_id, *args, **kwargs):
        """Sample a single feature from the data set.

        Parameters
        ----------
        feature_id: hashable
            name of the feature to sample. If no feature names were passed, then
            the features are accessible via their column index.

        """
        dm = DataManager(self[feature_id],
                         feature_names=[feature_id],
                         index=self.index)
        return dm.generate_sample(*args, **kwargs)


    def set_index(self, index):
        self.index = index
        if self.data_type in (pd.DataFrame, pd.Series):
            self.X.index = index


    def _labels_by_index(self, data_index):
        """ Method for grabbing labels associated with given indices.
        """
        # we coerce self.index to a list, so this is fine:
        numeric_index = [self.index.index(i) for i in data_index]

        # do we need to coerce labels to a particular data type?
        return self.y[numeric_index]


    @classmethod
    def _check_input(cls, dataset):
        """
        Ensures that dataset is pandas dataframe, and dataset is not empty
        :param dataset: skater.__datatypes__
        :return:
        """
        if not isinstance(dataset, (pd.DataFrame)):
            err_msg = "dataset must be a pandas.DataFrame"
            raise exceptions.DataSetError(err_msg)

        if len(dataset) == 0:
            err_msg = "dataset is empty"
            raise exceptions.DataSetError(err_msg)
