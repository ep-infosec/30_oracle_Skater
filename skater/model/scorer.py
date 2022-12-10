from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score, f1_score
from abc import ABCMeta, abstractmethod

from ..util.static_types import StaticTypes


class Scorer(object):
    """
    Base Class for all skater scoring functions.
    Any Scoring function must consume a model.
    Any scorer must determine the types of models that are compatible.

    """

    __metaclass__ = ABCMeta


    model_types = None
    prediction_types = None
    label_types = None

    def __init__(self, model):
        self.model = model

    @classmethod
    def check_params(cls):
        assert all([i in StaticTypes.model_types._valid_ for i in cls.model_types])
        assert all([i in StaticTypes.output_types._valid_ for i in cls.prediction_types])
        assert all([i in StaticTypes.output_types._valid_ for i in cls.label_types])


    @classmethod
    def check_model(cls, model):

        assert model.model_type in cls.model_types, "Scorer {0} not valid for models of type {1}, " \
                                                    "only {2}".format(cls,
                                                                      model.model_type,
                                                                      cls.model_types)


    def __call__(self, y_true, y_predicted, sample_weight=None):
        self.check_model(self.model)
        self.check_data(y_true, y_predicted)
        # formatted_y = self.model.transformer(self.model.output_formatter(y_true))
        return self._score(y_true, y_predicted, sample_weight=sample_weight)


    @staticmethod
    @abstractmethod
    def check_data(y_true, y_predicted):
        pass


class RegressionScorer(Scorer):
    model_types = [StaticTypes.model_types.regressor]
    prediction_types = [
        StaticTypes.output_types.numeric,
        StaticTypes.output_types.float,
        StaticTypes.output_types.int
    ]
    label_types = [
        StaticTypes.output_types.numeric,
        StaticTypes.output_types.float,
        StaticTypes.output_types.int
    ]

    @staticmethod
    def check_data(y_true, y_predicted):
        assert hasattr(y_predicted, 'shape'), \
            'outputs must have a shape attribute'
        assert hasattr(y_true, 'shape'), \
            'y_true must have a shape attribute'
        assert (len(y_predicted.shape) == 1) or (y_predicted.shape[1] == 1), \
            "Regression outputs must be 1D, " \
            "got {}".format(y_predicted.shape)
        assert (len(y_true.shape) == 1) or (y_true.shape[1] == 1), \
            "Regression outputs must be 1D, " \
            "got {}".format(y_true.shape)


# Regression Scorers
class MeanSquaredError(RegressionScorer):
    __name__ = "MSE"
    type = StaticTypes.scorer_types.decreasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None):
        return mean_squared_error(y_true, y_predicted, sample_weight=sample_weight)


class MeanAbsoluteError(RegressionScorer):
    __name__ = "MAE"
    type = StaticTypes.scorer_types.decreasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None):
        return mean_absolute_error(y_true, y_predicted, sample_weight=sample_weight)


class RSquared(RegressionScorer):
    __name__ = "R2"
    # Reference: https://en.wikipedia.org/wiki/Coefficient_of_determination
    # The score values range between [0, 1]. The best possible value is 1, however one could expect negative values as
    # well because of the arbitrary model fit.
    type = StaticTypes.scorer_types.increasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None):
        return r2_score(y_true, y_predicted, sample_weight=sample_weight)


class ClassifierScorer(Scorer):

    """
    * predictions must be N x K matrix with N rows and K classes.
    * labels must be be N x K matrix with N rows and K classes.
    """

    model_types = [StaticTypes.model_types.classifier]
    prediction_types = [StaticTypes.output_types.numeric, StaticTypes.output_types.float, StaticTypes.output_types.int]
    label_types = [StaticTypes.output_types.numeric, StaticTypes.output_types.float, StaticTypes.output_types.int]

    @staticmethod
    def check_data(y_true, y_predicted):
        assert hasattr(y_predicted, 'shape'), 'outputs must have a shape attribute'
        assert hasattr(y_true, 'shape'), 'y_true must have a shape attribute'


# Metrics related to Classification
class CrossEntropy(ClassifierScorer):
    __name__ = "cross-entropy"
    type = StaticTypes.scorer_types.decreasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None):
        """

        :param X: Dense X of probabilities, or binary indicator
        :param y:
        :param sample_weights:
        :return:
        """
        return log_loss(y_true, y_predicted, sample_weight=sample_weight)


class F1(ClassifierScorer):
    __name__ = "f1-score"
    type = StaticTypes.scorer_types.increasing

    @staticmethod
    def _score(y_true, y_predicted, sample_weight=None, average='weighted'):
        """

        :param X: Dense X of probabilities, or binary indicator
        :param y: indicator
        :param sample_weights:
        :return:
        """
        if len(y_predicted.shape) == 2:
            preds = y_predicted.argmax(axis=1)
        else:
            preds = y_predicted

        return f1_score(y_true, preds, sample_weight=sample_weight, average=average)


class ScorerFactory(object):
    """
    The idea is that we initialize the object with the model,
    but also provide an api for retrieving a static scoring function
    after checking that things are ok.
    """
    def __init__(self, model):
        if model.model_type == StaticTypes.model_types.regressor:
            self.mse = MeanSquaredError(model)
            self.mae = MeanAbsoluteError(model)
            self.r2 = RSquared(model)
            self.default = self.mae
        elif model.model_type == StaticTypes.model_types.classifier:
            self.cross_entropy = CrossEntropy(model)
            self.f1 = F1(model)
            # TODO: Not sure why the first condition is relevant, probably some early design decision.
            # TODO Need to check other examples and add more test before removing the first check completely
            if model.probability is not None and not 'unknown' or model.probability is True:
                self.default = self.cross_entropy
            else:
                self.default = self.f1

        self.type = self.default.type


    def __call__(self, y_true, y_predicted, sample_weight=None):
        return self.default(y_true, y_predicted, sample_weight=sample_weight)


    def get_scorer_function(self, scorer_type='default'):
        """
        Returns a scoring function as a pure function.

        Parameters
        ----------

        scorer_type: string
            Specifies which scorer to use. Default value 'default' returns f1 for classifiers that return labels,
            cross_entropy for classifiers that return probabilities, and mean absolute error for regressors.


        Returns
        -------
            .score staticmethod of skater.model.scorer.Scorer object.
        """
        assert scorer_type in self.__dict__, "Scorer type {} not recognized " \
                                             "or allowed for model type".format(scorer_type)
        scorer = self.__dict__[scorer_type]._score
        scorer.type = self.__dict__[scorer_type].type
        scorer.name = self.__dict__[scorer_type].__name__
        return scorer
