import pandas as pd
import numpy as np
import unittest
import sys
try:
    if sys.version_info >= (3, 5):
        from skater.core.global_interpretation.interpretable_models.brlc import BRLC
        from skater.core.validation import compute_validation_curve
except ImportError as err:
    # Because of unresolved error related to ImportError: dlopen: cannot load any more object with static TLS(#229)
    raise unittest.SkipTest(str(err))


# When in Dev mode, a consistent mode to validate test, that would keep track of weird segmentation fault is using gdb
# (GNU Debugger). This is a temporary workaround. Follow the below mentioned steps
# 1. sudo apt install gdb
# 2. gdb python
# 3. r skater/tests/test_rule_list.py
# 4. Result: All the tests should succeed, and if it fails it will point out the trace for failure
@unittest.skipIf(sys.version_info < (3, 5), "only Bayesian Rule List supported only for python 3.5/3.6")
class TestRuleList(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sbrl_inst = BRLC(min_rule_len=1, max_rule_len=2, iterations=10000, n_chains=3)
        cls.input_data = pd.read_csv('skater/tests/data/sample_data.csv')
        # data transformation and cleaning ...
        cls.input_data["Sex"] = cls.input_data["Sex"].astype('category')
        cls.input_data["Sex_Encoded"] = cls.input_data["Sex"].cat.codes
        cls.input_data["Embarked"] = cls.input_data["Embarked"].astype('category')
        cls.input_data["Embarked_Encoded"] = cls.input_data["Embarked"].cat.codes
        cls.input_data = cls.input_data.drop(['Ticket', 'Cabin', 'Name', 'Sex', 'Embarked'], axis=1)
        # Remove NaN values
        cls.input_data = cls.input_data.dropna()
        cls.y = cls.input_data['Survived']
        cls.input_data = cls.input_data.drop(['Survived'], axis=1)
        # Train a model
        cls.sbrl_inst.fit(cls.input_data[1:50], cls.y[1:50], undiscretize_feature_list=["PassengerId", "Pclass",
                                                                                        "SibSp", "Parch",
                                                                                        "Sex_Encoded",
                                                                                        "Embarked_Encoded"])


    def test_discretizer(self):
        new_df = self.sbrl_inst.discretizer(self.input_data, column_list=["Age"])
        self.assertEquals(new_df["Age_q_label"].shape[0] > 0, True)


    def test_model_build(self):
        new_data = self.sbrl_inst.discretizer(self.input_data, column_list=["Age", "Fare"])
        result_score = self.sbrl_inst.predict_proba(new_data)
        result_labels = self.sbrl_inst.predict(new_data)

        # make sure shape of the data-frame is as expected
        self.assertEquals(result_score.shape, (77, 2))
        self.assertEquals(result_labels[1].shape, (77, ))

        generated_labels = np.unique(result_labels[1])
        expected_labels = np.array([0, 1])
        self.assertEquals(np.array_equal(generated_labels, expected_labels), True)


    def test_model_save_load(self):
        self.sbrl_inst.save_model("test.pkl", compress=True)
        # Explicitly assigning the model instance to 'None' to validate loading of persisted model
        # Care is advised when handing the model instance, it might make the model unstable
        self.sbrl_inst.model = None
        self.assertEquals(self.sbrl_inst.model is None, True)
        self.sbrl_inst.load_model("test.pkl")
        self.assertEquals(self.sbrl_inst.model is not None, True)


    def test_model_output(self):
        result = self.sbrl_inst.access_learned_rules('23:25')
        self.assertEquals(len(result), 2)


    @unittest.skip("Support for computing validation curve for SBRL is still under development")
    def test_validation(self):
        param_range = [3, 4]
        train_scores, test_scores = compute_validation_curve(self.sbrl_inst, n_folds=2, x=self.input_data, y=self.y,
                                                             param_name="rule_minlen", param_range=param_range)
        self.assertEquals(train_scores.shape[0], 2)
        self.assertEquals(test_scores.shape[0], 2)


if __name__ == '__main__':
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRuleList)
        unittest.TextTestRunner(verbosity=2).run(suite)
