"""Base ModelInterpreter class. submodules like lime and partial dependence"
must have these methods"""


class ModelInterpreter(object):
    """
    Base Interpreter class. Common methods include loading a dataset and type setting.
    """

    def __init__(self, interpreter):
        self.interpreter = interpreter

    @property
    def data_set(self):
        """data_set routes to the Interpreter's dataset"""
        return self.interpreter.data_set

    @property
    def training_labels(self):
        """training_labels routes to the Interpreter's training labels"""
        return self.interpreter.training_labels

    @staticmethod
    def _types():
        return ['partial_dependence', 'lime']

    def load_data(self, training_data, index=None, feature_names=None):
        """.consider routes to Interpreter's .consider"""
        self.interpreter.consider(training_data, index=index, feature_names=feature_names)
