import unittest

import pandas as pd
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection._split import train_test_split

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from skater.util.logger import _INFO


class TestTreeSurrogates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Classification use-case
        cls.X_c, cls.y_c = make_moons(1000, noise=0.5)
        cls.X_c = pd.DataFrame(cls.X_c, columns=['F1', 'F2'])
        cls.target_names = ['class 0', 'class 1']
        cls.X_train_c, cls.X_test_c, cls.y_train_c, cls.y_test_c = train_test_split(cls.X_c, cls.y_c)
        cls.classifier_est = DecisionTreeClassifier(max_depth=5, random_state=5)
        cls.classifier_est.fit(cls.X_train_c, cls.y_train_c)
        cls.interpreter = Interpretation(cls.X_train_c, feature_names=cls.X_c.columns)
        cls.model_inst = InMemoryModel(cls.classifier_est.predict, examples=cls.X_train_c,
                                       model_type='classifier', unique_values=[0, 1], feature_names=cls.X_c.columns,
                                       target_names=cls.target_names, log_level=_INFO)

    # all the below tests are with F1-score
    def test_surrogate_no_pruning(self):
        surrogate_explainer = self.interpreter.tree_surrogate(oracle=self.model_inst, seed=5)
        result = surrogate_explainer.fit(self.X_train_c, self.y_train_c, use_oracle=True,
                                         prune=None, scorer_type='default')
        self.assertEquals(result < 0, True)


    def test_surrogate_with_prepruning(self):
        surrogate_explainer = self.interpreter.tree_surrogate(oracle=self.model_inst, seed=5)
        result = surrogate_explainer.fit(self.X_train_c, self.y_train_c, use_oracle=True,
                                         prune='pre', scorer_type='f1')
        self.assertEquals(result < 0, True)


    def test_surrogate_with_postpruning(self):
        surrogate_explainer = self.interpreter.tree_surrogate(oracle=self.model_inst, seed=5)
        result = surrogate_explainer.fit(self.X_train_c, self.y_train_c, use_oracle=True,
                                         prune='post', scorer_type='f1')
        self.assertEquals(result < 0, True)


    def test_surrogate_with_cross_entropy(self):
        model_inst = InMemoryModel(self.classifier_est.predict_proba, examples=self.X_train_c,
                                   model_type='classifier', feature_names=self.X_c.columns,
                                   target_names=self.target_names, log_level=_INFO, probability=True)
        surrogate_explainer = self.interpreter.tree_surrogate(oracle=model_inst, seed=5)
        result = surrogate_explainer.fit(self.X_train_c, self.y_train_c, use_oracle=True,
                                         prune='post', scorer_type='default')
        self.assertEqual(surrogate_explainer.scorer_name_, 'cross-entropy', True)
        self.assertEquals(result != 0, True)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTreeSurrogates)
    unittest.TextTestRunner(verbosity=2).run(suite)
