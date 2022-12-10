from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from skater.model import InMemoryModel
from skater.util.static_types import StaticTypes
from skater.tests.arg_parser import arg_parse, create_parser
import unittest


class TestModel(unittest.TestCase):
    """
    Tests the skater.model.* types
    """

    def setUp(self):
        """Create data for testing"""
        self.parser = create_parser()
        args = self.parser.parse_args()
        debug = args.debug

        if debug:
            self.log_level = 10
        else:
            self.log_level = 30


    def test_issues_161_and_189(self):
        """
        ensure DataManager(data).data == data
        """
        X, y = load_breast_cancer(True)
        X, y = X[15:40], y[15:40]
        model = KNeighborsClassifier(weights='distance', p=2, n_neighbors=10).fit(X, y)
        skater_model = InMemoryModel(model.predict_proba, examples=X, probability=True)
        assert skater_model.probability is True
        assert skater_model.model_type == StaticTypes.model_types.classifier


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(unittest.makeSuite(TestModel))
