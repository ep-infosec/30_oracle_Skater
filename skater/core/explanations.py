"""Interpretation Class"""
from skater.core.global_interpretation.partial_dependence import PartialDependence
from skater.core.global_interpretation.feature_importance import FeatureImportance
from skater.core.global_interpretation.tree_surrogate import TreeSurrogate
from skater.data import DataManager
from skater.util.logger import build_logger


class Interpretation(object):
    """
    Interpretation class. Before calling interpretation subclasses like partial
    dependence, one must call Interpretation.load_data().


    Examples
    --------
        >>> from skater.core.explanations import Interpretation
        >>> interpreter = Interpretation()
        >>> interpreter.load_data(X, feature_ids = ['a','b'])
        >>> interpreter.partial_dependence([feature_id1, feature_id2], regressor.predict)
    """

    def __init__(self, training_data=None, training_labels=None, class_names=None, feature_names=None, index=None,
                 log_level=30):
        """
        Attaches local and global interpretations
        to Interpretation object.

        Parameters
        -----------
        log_level: int
            Logger Verbosity, see https://docs.python.org/2/library/logging.html
            for details.

        """
        self._log_level = log_level
        self.logger = build_logger(log_level, __name__)
        self.data_set = None
        self.feature_names = feature_names
        self.class_names = class_names
        self.load_data(training_data,
                       training_labels=training_labels,
                       feature_names=feature_names,
                       index=index)
        self.partial_dependence = PartialDependence(self)
        self.feature_importance = FeatureImportance(self)
        self.tree_surrogate = TreeSurrogate


    def load_data(self, training_data, training_labels=None, feature_names=None, index=None):
        """
        Creates a DataSet object from inputs, ties to interpretation object.
        This will be exposed to all submodules.

        Parameters
        ----------
        training_data: numpy.ndarray, pandas.DataFrame
            the dataset. can be 1D or 2D

        feature_names: array-type
            names to call features.

        index: array-type
            names to call rows.


        Returns
        --------
            None
        """

        self.logger.info("Loading Data")
        self.data_set = DataManager(training_data,
                                    y=training_labels,
                                    feature_names=feature_names,
                                    index=index,
                                    log_level=self._log_level)
        self.logger.info("Data loaded")
        self.logger.info("Data shape: {}".format(self.data_set.X.shape))
        self.logger.info("Dataset Feature_ids: {}".format(self.data_set.feature_ids))
