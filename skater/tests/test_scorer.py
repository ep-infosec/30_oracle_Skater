import unittest

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection._split import train_test_split

from skater.model import InMemoryModel


class TestScorer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X, y = make_moons(1000, noise=0.5)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y)
        cls.classifier_est = DecisionTreeClassifier(max_depth=5)
        cls.classifier_est.fit(cls.X_train, cls.y_train)


    def test_compute_default_scores(self):
        # For classification default scorer is weighted F1-score
        model_inst = InMemoryModel(self.classifier_est.predict, examples=self.X_train,
                                   model_type='classifier', unique_values=[0, 1, 2])
        scorer = model_inst.scorers.get_scorer_function(scorer_type='default')
        self.assertEqual(scorer.name == 'f1-score', True)

        scorer = model_inst.scorers.get_scorer_function(scorer_type='f1')
        self.assertEqual(scorer.name == 'f1-score', True)

        y_hat = self.classifier_est.predict(self.X_test)
        value = scorer(self.y_test, y_hat, average='weighted')
        self.assertEquals(value > 0, True)


    def test_compute_log_loss(self):
        model_inst = InMemoryModel(self.classifier_est.predict_proba, examples=self.X_train, probability=True,
                                   model_type='classifier')
        scorer = model_inst.scorers.get_scorer_function(scorer_type='default')
        self.assertEqual(scorer.name == 'cross-entropy', True)

        scorer = model_inst.scorers.get_scorer_function(scorer_type='cross_entropy')
        self.assertEqual(scorer.name == 'cross-entropy', True)

        y_hat = self.classifier_est.predict_proba(self.X_test)
        value = scorer(self.y_test, y_hat)
        self.assertEquals(value > 0, True)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScorer)
    unittest.TextTestRunner(verbosity=2).run(suite)
