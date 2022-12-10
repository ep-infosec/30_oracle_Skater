import unittest

import numpy as np
from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from functools import partial

from skater.core.explanations import Interpretation
from skater.util import exceptions
from .arg_parser import create_parser
from skater.model import InMemoryModel, DeployedModel
from skater.util.dataops import MultiColumnLabelBinarizer
from skater.core.global_interpretation.partial_dependence import PartialDependence


class TestPartialDependence(unittest.TestCase):

    def setUp(self):
        args = create_parser().parse_args()
        debug = args.debug
        self.seed = args.seed
        self.n = args.n
        self.dim = args.dim
        self.features = [str(i) for i in range(self.dim)]
        self.X = norm.rvs(0, 1, size=(self.n, self.dim), random_state=self.seed)
        self.B = np.array([-10.1, 2.2, 6.1])
        self.y = np.dot(self.X, self.B)
        self.y_as_int = np.round(expit(self.y))
        self.y_as_string = np.array([str(i) for i in self.y_as_int])
        # example dataset for y = B.X
        # X = array([[ 1.62434536, -0.61175641, -0.52817175], ... [-0.15065961, -1.40002289, -1.30106608]])  (1000 * 3)
        # B = array([-10.1,   2.2,   6.1])
        # y = array([ -2.09736000e+01,  -1.29850618e+00,  -1.73511155e+01, ...]) (1000 * 1)
        # features = ['0', '1', '2']
        ##
        # Other output types:
        # y_as_int = array[ 0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1., ...]
        # y_as_string = array['0.0', '0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', ... ]


        # Another set of input
        # sample data
        self.sample_x = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
        self.sample_y = np.array([-1, -1, -1, 1, 1, 1])
        self.sample_feature_name = [str(i) for i in range(self.sample_x.shape[1])]

        if debug:
            self.interpreter = Interpretation(self.X, feature_names=self.features, log_level='DEBUG')
        else:
            self.interpreter = Interpretation(self.X, feature_names=self.features)  # default level is 'WARNING'

        self.regressor = LinearRegression()
        self.regressor.fit(self.X, self.y)
        self.regressor_predict_fn = InMemoryModel(self.regressor.predict, examples=self.X)

        self.classifier = LogisticRegression()
        self.classifier.fit(self.X, self.y_as_int)
        self.classifier_predict_fn = InMemoryModel(self.classifier.predict, examples=self.X, unique_values=self.classifier.classes_)
        self.classifier_predict_proba_fn = InMemoryModel(self.classifier.predict_proba, examples=self.X)

        self.string_classifier = LogisticRegression()
        self.string_classifier.fit(self.X, self.y_as_string)
        self.string_classifier_predict_fn = InMemoryModel(self.string_classifier.predict_proba, examples=self.X)


        # Yet another set of input!!
        self.sample_x_categorical = np.array([['B', -1], ['A', -1], ['A', -2], ['C', 1], ['C', 2], ['A', 1]])
        self.sample_y_categorical = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        self.categorical_feature_names = ['Letters', 'Numbers']
        self.categorical_transformer = MultiColumnLabelBinarizer()
        self.categorical_transformer.fit(self.sample_x_categorical)
        self.sample_x_categorical_transormed = self.categorical_transformer.transform(self.sample_x_categorical)
        self.categorical_classifier = LogisticRegression()
        self.categorical_classifier.fit(self.sample_x_categorical_transormed, self.sample_y_categorical)
        self.categorical_predict_fn = lambda x: self.categorical_classifier.predict_proba(self.categorical_transformer.transform(x))
        self.categorical_model = InMemoryModel(self.categorical_predict_fn, examples=self.sample_x_categorical)


    def test_pdp_with_default_sampling(self):
        pdp_df = self.interpreter.partial_dependence.partial_dependence([self.features[0]],
                                                                        self.regressor_predict_fn,
                                                                        sample=True)
        self.assertEquals(pdp_df.shape, (30, 4))  # default grid resolution is 30


    def test_pd_with_categorical_features(self):
        interpreter = Interpretation(self.sample_x_categorical, feature_names=self.categorical_feature_names)
        try:
            interpreter.partial_dependence.partial_dependence([self.categorical_feature_names[0]], self.categorical_model)
        except:
            self.fail("PD computation function failed with categorical features")


    def test_partial_dependence_binary_classification(self):
        # In the default implementation of pdp on sklearn, there is an approx. done
        # if the number of unique values for a feature space < grid_resolution specified.
        # For now, we have decided to not have that approximation. In V2, we will be benchmarking for
        # performance as well. Around that time we will revisit the same.
        # Reference: https://github.com/scikit-learn/scikit-learn/blob/4d9a12d175a38f2bcb720389ad2213f71a3d7697/sklearn/ensemble/tests/test_partial_dependence.py
        # TODO: check on the feature space approximation (V2)
        # Test partial dependence for classifier
        clf = GradientBoostingClassifier(n_estimators=10, random_state=1)
        clf.fit(self.sample_x, self.sample_y)
        classifier_predict_fn = InMemoryModel(clf.predict_proba, examples=self.sample_x)
        interpreter = Interpretation(training_data=np.array(self.sample_x), feature_names=self.sample_feature_name)
        pdp_df = interpreter.partial_dependence.partial_dependence(['0'],
                                                                   classifier_predict_fn,
                                                                   grid_resolution=5,
                                                                   sample=True)

        self.assertEquals(pdp_df.shape[0], len(np.unique(interpreter.data_set['0'])))

        # now with our own grid
        ud_grid = np.unique(self.sample_x[:, 0])
        # input: array([-2, -1,  1,  2])
        # the returned grid should have only 4 values as specified by the user
        pdp_df = interpreter.partial_dependence.partial_dependence(['0'], classifier_predict_fn,
                                                                   grid=ud_grid, sample=True)
        self.assertEquals(pdp_df.shape[0], 4)


    def test_partial_dependence_multiclass(self):
        # Iris data classes: ['setosa', 'versicolor', 'virginica']
        iris = datasets.load_iris()
        # 1. Using GB Classifier
        clf = GradientBoostingClassifier(n_estimators=10, random_state=1)
        clf.fit(iris.data, iris.target)
        classifier_predict_fn = InMemoryModel(clf.predict_proba, examples=iris.data)
        interpreter = Interpretation(training_data=iris.data, feature_names=iris.feature_names)
        pdp_df = interpreter.partial_dependence.partial_dependence([iris.feature_names[0]], classifier_predict_fn,
                                                                   grid_resolution=25, sample=True)

        expected_feature_name = PartialDependence.feature_column_name_formatter('sepal length (cm)')

        self.assertIn(expected_feature_name,
                      pdp_df.columns.values,
                      "{0} not in columns {1}".format(expected_feature_name,
                                                      pdp_df.columns.values))
        # 2. Using SVC
        from sklearn import svm
        # With SVC, predict_proba is supported only if probability flag is enabled, by default it is false
        clf = svm.SVC(probability=True)
        clf.fit(iris.data, iris.target)
        classifier_predict_fn = InMemoryModel(clf.predict_proba, examples=iris.data)
        interpreter = Interpretation(training_data=iris.data, feature_names=iris.feature_names)

        pdp_df = interpreter.partial_dependence.partial_dependence([iris.feature_names[0]], classifier_predict_fn,
                                                                   grid_resolution=25, sample=True)
        self.assertIn(expected_feature_name,
                      pdp_df.columns.values,
                      "{} not in columns {}".format(*[expected_feature_name,
                                                      pdp_df.columns.values]))


    def test_pdp_regression_coefs_closeness(self, epsilon=1):
        pdp_df = self.interpreter.partial_dependence.partial_dependence([self.features[0]],
                                                                        self.regressor_predict_fn)
        val_col = PartialDependence.feature_column_name_formatter(self.features[0])

        y = np.array(pdp_df[self.regressor_predict_fn.target_names[0]])
        x = np.array(pdp_df[val_col])[:, np.newaxis]
        regressor = LinearRegression()
        regressor.fit(x, y)
        self.interpreter.logger.debug("Regressor coefs: {}".format(regressor.coef_))
        self.interpreter.logger.debug("Regressor coef shape: {}".format(regressor.coef_.shape))
        coef = regressor.coef_[0]
        self.assertTrue(abs(coef - self.B[0]) < epsilon, True)


    def test_2D_pdp(self):
        try:
            self.interpreter.partial_dependence.partial_dependence(self.features[:2],
                                                                   self.regressor_predict_fn,
                                                                   grid_resolution=10,
                                                                   sample=True)
        except:
            self.fail("2D regressor pd failed")


    @unittest.skip("Related to plotting ...")
    def test_plot_1D_pdp(self):
        try:
            self.interpreter.partial_dependence.plot_partial_dependence([self.features[0]],
                                                                        self.regressor_predict_fn,
                                                                        grid_resolution=10)
        except:
            self.fail("1D regressor plot failed")


    @unittest.skip("Related to plotting ...")
    def test_plot_1D_pdp_with_sampling(self):
        try:
            self.interpreter.partial_dependence.plot_partial_dependence(
                [self.features[0]],
                self.regressor_predict_fn,
                grid_resolution=10,
                sample=True)
        except:
            self.fail("1D classifier plot with sampling failed")


    @unittest.skip("Related to plotting ...")
    def test_plot_2D_pdp(self):
        try:
            self.interpreter.partial_dependence.plot_partial_dependence(self.features[:2],
                                                                        self.regressor_predict_fn,
                                                                        grid_resolution=10,
                                                                        sample=False)
        except:
            self.fail("2D partial dep plot failed")


    @unittest.skip("Related to plotting ...")
    def test_plot_2D_pdp_with_sampling(self):
        try:
            self.interpreter.partial_dependence.plot_partial_dependence(self.features[:2],
                                                                        self.regressor_predict_fn,
                                                                        grid_resolution=10,
                                                                        sample=True)
        except:
            self.fail("2D regressor with sampling failed")


    def test_fail_when_grid_range_is_outside_0_and_1(self):
        pdp_func = partial(self.interpreter.partial_dependence.partial_dependence,
                           *[[self.features[0]], self.regressor_predict_fn],
                           **{'grid_range': (.01, 1.01)})
        self.assertRaises(exceptions.MalformedGridRangeError, pdp_func)


    def test_pdp_1d_classifier_no_proba(self):
        self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                               self.classifier_predict_fn,
                                                               grid_resolution=10)
        try:
            self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                                   self.classifier_predict_fn,
                                                                   grid_resolution=10)
        except:
            self.fail("1D pdp without proba failed")


    def test_pdp_2d_classifier_no_proba(self):
        try:
            self.interpreter.partial_dependence.partial_dependence(self.features[:2],
                                                                   self.classifier_predict_fn,
                                                                   grid_resolution=10)
        except:
            self.fail("2D pdp without proba failed")


    def test_pdp_1d_classifier_with_proba(self):
        try:
            self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                                   self.classifier_predict_proba_fn,
                                                                   grid_resolution=10)
        except:
            self.fail("1D classifier with probability scores failed")


    def test_pdp_2d_classifier_with_proba(self):
        try:
            self.interpreter.partial_dependence.partial_dependence(self.features[:2],
                                                                   self.classifier_predict_proba_fn,
                                                                   grid_resolution=10)
        except:
            self.fail("2D classifier with probability scores failed")


    def test_pdp_1d_string_classifier_no_proba(self):
        def fail_func():
            self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                                   InMemoryModel(self.string_classifier.predict,
                                                                                 examples=self.X),
                                                                   grid_resolution=10)
        self.assertRaises(exceptions.ModelError, fail_func)


    def test_pdp_1d_string_classifier_with_proba(self):
        try:
            self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                                   self.string_classifier_predict_fn,
                                                                   grid_resolution=10)
        except:
            self.fail('1D string classifier pd failed')


    def test_pd_with_long_string_feature_name_issue_166(self):

        feature_names = ['longstring_{}'.format(i) for i in range(self.X.shape[1])]
        interpreter = Interpretation(self.X, feature_names=feature_names)
        try:
            interpreter.partial_dependence.partial_dependence(feature_names[0],
                                                              self.regressor_predict_fn)
        except:
            self.fail("1D Partial dependence failed when passing long string name")


    def test_pd_with_variance_type_prediction(self):
        try:
            self.interpreter.partial_dependence.partial_dependence(self.features[:1],
                                                                   self.classifier_predict_fn,
                                                                   grid_resolution=10,
                                                                   variance_type='prediction')
        except:
            self.fail("pd with Variance Type=prediction failed")


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPartialDependence)
    unittest.TextTestRunner(verbosity=2).run(suite)
