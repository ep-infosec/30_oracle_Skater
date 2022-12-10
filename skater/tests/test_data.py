import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from skater.data import DataManager
from .arg_parser import arg_parse, create_parser


class TestData(unittest.TestCase):
    """
    Tests the skater.data.DataManager object
    """

    def setUp(self):
        """Create data for testing"""
        self.parser = create_parser()
        args = self.parser.parse_args()
        debug = args.debug

        self.dim = 3
        self.n = 8

        self.X = np.array([
            [0, 0, 0],
            [0, 0, 2],
            [0, 1, 0],
            [0, 2, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [2, 1, 1],
        ])

        self.feature_names = ['x{}'.format(i) for i in range(self.dim)]
        self.index = ['{}'.format(i) for i in range(self.n)]

        if debug:
            self.log_level = 10
        else:
            self.log_level = 30


    def test_datamanager_data_returns_original(self):
        """
        ensure DataManager(data).data == data
        """
        data_set = DataManager(self.X,
                               feature_names=self.feature_names,
                               index=self.index)
        assert_array_equal(data_set.X, self.X)


    def test_1d_numpy_array(self):
        """
        ensure DataManager(data) works when data is 1D np.ndarray
        """

        feature_id = 0
        array_1d = self.X[:, feature_id][:, np.newaxis]
        feature_names = [self.feature_names[feature_id]]

        data_set = DataManager(array_1d,
                               feature_names=feature_names,
                               index=self.index,
                               log_level=self.log_level)
        assert_array_equal(data_set.X, array_1d)
        assert data_set.feature_ids == feature_names


    def test_2d_numpy_array(self):
        """
        ensure DataManager(data) works when data is 2D np.ndarray
        """

        feature_ids = [0, 1]
        array_2d = self.X[:, feature_ids]
        feature_names = [self.feature_names[i] for i in feature_ids]

        data_set = DataManager(array_2d,
                               feature_names=feature_names,
                               index=self.index,
                               log_level=self.log_level)
        assert_array_equal(data_set.X, array_2d)
        assert data_set.feature_ids == feature_names


    def test_pandas_dataframe(self):
        """
        Ensure DataManager(data) works when data is pd.DataFrame
        """
        X_as_dataframe = pd.DataFrame(self.X, columns=self.feature_names, index=self.index)
        data_set = DataManager(X_as_dataframe,
                               log_level=self.log_level,
                               index=self.index,
                               feature_names=self.feature_names)

        assert data_set.feature_ids == self.feature_names, "Feature Names from DataFrame " \
                                                           "not loaded properly"
        assert data_set.index == self.index, "Index from DataFrame not loaded properly"
        assert_array_equal(data_set.X, self.X)
        try:
            feature_info = data_set.feature_info
            for feature in self.feature_names:
                assert feature in feature_info
        except ValueError as e:
            raise e
        except KeyError as e:
            assert False, "{:} not found in feature_info".format(feature)


    def test_generate_grid_1_variable(self):
        """Ensures generate grid works with 1 variable"""
        data_set = DataManager(self.X, feature_names=self.feature_names, index=self.index)
        grid = data_set.generate_grid(data_set.feature_ids[0:1], grid_resolution=100)
        self.assertEquals(len(grid), 1)


    def test_generate_grid_2_variables(self):
        """Ensures generate grid works with 2 variables"""
        data_set = DataManager(self.X, feature_names=self.feature_names, index=self.index)
        grid = data_set.generate_grid(self.feature_names[0:2], grid_resolution=100)
        self.assertEquals(len(grid), 2)


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(unittest.makeSuite(TestData))
