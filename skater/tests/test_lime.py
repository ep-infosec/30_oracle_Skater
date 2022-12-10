import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from functools import partial

from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer


class TestLime(unittest.TestCase):
    """
    Test imported lime package
    """

    def setUp(self):
        """
        Build data for testing
        :param n:
        :param dim:
        :return:
        """

        self.X = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ])
        self.n, self.dim = self.X.shape
        self.feature_names = ['x{}'.format(i) for i in range(self.dim)]
        self.index = ['{}'.format(i) for i in range(self.n)]

        self.B = np.array([-5, 0, 5])
        self.y = np.dot(self.X, self.B) + np.random.normal(0, .01, size=self.n)
        self.y_for_classifier = np.round(expit(self.y))
        self.example = self.X[0]

        self.seed = 1
        self.regressor = LinearRegression()
        self.regressor.fit(self.X, self.y)

        self.classifier = LogisticRegression()
        self.classifier.fit(self.X, self.y_for_classifier)

        self.model_regressor = LinearRegression()

    def test_regression_with_feature_names(self):
        """
        Ensure lime.lime_tabular works when predict_fn = regressor.predict
        and feature names are passed
        :return:
        """

        interpretor = LimeTabularExplainer(self.X, feature_names=self.feature_names, mode="regression")
        assert interpretor.explain_instance(self.example, self.regressor.predict)

    def test_regression_without_feature_names(self):
        """
        Ensure lime.lime_tabular works when predict_fn = regressor.predict
        and feature names are NOT passed
        :return:
        """
        interpretor = LimeTabularExplainer(self.X, mode="regression")
        assert interpretor.explain_instance(self.example, self.regressor.predict)

    def test_classifier_no_proba_without_feature_names(self):
        """
        Ensure lime.lime_tabular works when predict_fn = classifier.predict
        and feature names are NOT passed
        :return:
        """

        interpretor = LimeTabularExplainer(self.X)
        interpretor_func = partial(interpretor.explain_instance, *[self.example, self.classifier.predict])
        self.assertRaises(NotImplementedError, interpretor_func)

    def test_classifier_with_proba_without_feature_names(self):
        """
        Ensure lime.lime_tabular works when predict_fn = classifier.predict_proba
        and feature names are NOT passed
        :return:
        """

        interpretor = LimeTabularExplainer(self.X)
        assert interpretor.explain_instance(self.example, self.classifier.predict_proba)

    def test_classifier_no_proba_with_feature_names(self):
        """
        Ensure lime.lime_tabular works when predict_fn = classifier.predict
        and feature names are passed
        :return:
        """

        interpretor = LimeTabularExplainer(self.X, feature_names=self.feature_names)
        interpretor_func = partial(interpretor.explain_instance, *[self.example, self.classifier.predict])
        self.assertRaises(NotImplementedError, interpretor_func)

    def test_classifier_with_proba_with_feature_names(self):
        """
        Ensure lime.lime_tabular works when predict_fn = classifier.predict_proba
        and feature names are passed
        :return:
        """

        interpretor = LimeTabularExplainer(self.X, feature_names=self.feature_names)
        assert interpretor.explain_instance(self.example, self.classifier.predict_proba)

    def test_lime_coef_accuracy(self):
        """
        Ensure that for a trivial example, the coefficients of a regressor explanation
        are all similar to the true beta values of the generative process.

        :return:
        """

        error_epsilon = .1
        explainer = LimeTabularExplainer(self.X,
                                         discretize_continuous=True, mode="regression")
        explanation = explainer.explain_instance(self.example,
                                                 self.regressor.predict,
                                                 model_regressor=self.model_regressor)

        vals = dict(explanation.as_list())
        keys = ['{} <= 0.00'.format(i) for i in [2, 1, 0]]
        lime_coefs = np.array([vals[key] for key in keys])
        assert (abs(self.regressor.coef_ - lime_coefs) < error_epsilon).all()



if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(unittest.makeSuite(TestLime))
