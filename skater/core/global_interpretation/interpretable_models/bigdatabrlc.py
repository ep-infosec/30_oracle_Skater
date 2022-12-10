# -*- coding: UTF-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

from skater.core.global_interpretation.interpretable_models.brlc import BRLC
from skater.util import exceptions


class BigDataBRLC(BRLC):
    """
    :: Experimental :: The implementation is currently experimental and might change in future

    BigDataBRLC is a BRLC to handle large data-sets. Advisable to be used when the number of
    input examples>1k. It approximates large datasets with the help of surrogate(metamodel) estimators. For example, it uses
    surrogate estimator such as SVC(Support Vector Classifier) or RandomForest by default to filter the data
    points which are closest to the decision boundary. The idea is to identify the minimum training set size
    (controlled by the parameter sub_sample_percentage) with the goal to maximize accuracy.
    This helps in reducing the computation time to build the final BRL.

    Parameters
    ----------
    sub_sample_percentage : float (default=0.1)
        specify the fraction of the training sample to be retained for training BRL.

    iterations : int (default=30000)
        number of iterations for each MCMC chain.

    pos_sign : int (default=1)
        sign for the positive labels in the "label" column.

    neg_sign : int (default=0)
        sign for the negative labels in the "label" column.

    min_rule_len : int (default=1)
        minimum number of cardinality for rules to be mined from the data-frame.

    max_rule_len : int (default=8)
        maximum number of cardinality for rules to be mined from the data-frame.

    min_support_pos : float (default=0.1)
        a number between 0 and 1, for the minimum percentage support for the positive observations.

    min_support_neg : float (default 0.1)
        a number between 0 and 1, for the minimum percentage support for the negative observations.

    eta : int (default=1)

    n_chains: int (default=10)

    alpha : int (default=1)
        a prior pseudo-count for the positive(alpha1) and negative(alpha0) classes. Default values (1, 1)

    lambda_ : int (default=8)
        a hyper-parameter for the expected length of the rule list.

    discretize : bool (default=True)
        apply discretizer to handle continuous features.

    drop_features : bool (default=False)
        once continuous features are discretized, use this flag to either retain or drop them from the dataframe

    threshold : float (default=0.5)
        specify the threshold for the decision boundary. This is the probability level to compute \
        distance of the predictions(for input examples) from the decision boundary. Input examples closest to the \
        decision boundary are sub-sampled. Size of sub-sampled data is controlled using 'sub_sample_percentage'.

    penalty_param_svm : float (default=0.01)
        Regularization parameter('C') for Linear Support Vector Classifier. Lower regularization value forces the \
        optimizer to maximize the hyperplane.

        References: https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel

    calibration_type : string (default='sigmoid')
        Calibrate the base estimator's prediction(currently, all the base estimators are calibrated, that might
        change in future with more experimentation). Calibration could be performed in 2 ways
        1. parametric approach using Platt Scaling ('sigmoid')
        2. non-parametric approach using isotonic regression('isotonic).
        Avoid using isotonic regression for input examples<<1k because it tends to over-fit.

        References:

        [1] A. Niculescu-Mizil & R. Caruana(ICML2005), Predicting Good Probabilities With Supervised Learning

        [2] https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf

        [3] http://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/

    cv_calibration : int (default=3)
        specify number of folds for cross-validation splitting strategy

    random_state: int (default=0)
    surrogate_estimator: string (default='SVM', 'RF': RandomForest)
        Surrogate model to build the initial model for handling large datasets. Currently, SVM and RandomForest
        is supported.

    References
    ----------

    .. [1] Dr. Tamas Madl, https://github.com/tmadl/sklearn-expertsys/blob/master/BigDataRuleListClassifier.py
    .. [2] https://pdfs.semanticscholar.org/e44c/9dcf90d5a9a7e74a1d74c9900ff69142c67f.pdf
    .. [3] Surrogate model: https://en.wikipedia.org/wiki/Surrogate_model
    .. [4] W. Andrew Pruett , Robert L. Hester(2016),
       The Creation of Surrogate Models for Fast Estimation of Complex Model Outcomes
       (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0156574)

    Examples
    --------
    >>> from skater.core.global_interpretation.interpretable_models.brlc import BRLC
    >>> from skater.core.global_interpretation.interpretable_models.bigdatabrlc import BigDataBRLC
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    ...
    >>> Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
    >>> input_df = pd.read_csv('input_data.csv', skiprows=1)
    >>> sbrl_big = BigDataBRLC(sub_sample_percentage=0.1, min_rule_len=1, max_rule_len=3, iterations=10000,
    ...                                                 n_chains=3, surrogate_estimator="SVM", drop_features=True)
    >>> n_x, n_y = sbrl_big.subsample(Xtrain, ytrain, pos_label=1)
    >>> model = sbrl_big.fit(n_x, n_y, bin_labels='default')
    # For a complete example refer to credit_analysis_rule_lists.ipynb notebook in the `examples` section
    """
    def __init__(self, sub_sample_percentage=0.1, iterations=30000, pos_sign=1, neg_sign=0, min_rule_len=1,
                 max_rule_len=8, min_support_pos=0.10, min_support_neg=0.10, eta=1.0, n_chains=10, alpha=1,
                 lambda_=8, discretize=True, drop_features=False, threshold=0.5, penalty_param_svm=0.01,
                 calibration_type='sigmoid', cv_calibration=3, random_state=0, surrogate_estimator='SVM'):

        self.sample_percentage = sub_sample_percentage
        self.threhold = threshold

        self.surrogate_estimator = CalibratedClassifierCV(RandomForestClassifier(warm_start=True, oob_score=True,
                                                                                 max_features="sqrt",
                                                                                 random_state=random_state),
                                                          method=calibration_type, cv=cv_calibration) \
            if surrogate_estimator is 'RF' else surrogate_estimator

        self.surrogate_estimator = CalibratedClassifierCV(LinearSVC(C=penalty_param_svm, random_state=random_state),
                                                          method=calibration_type, cv=cv_calibration) \
            if surrogate_estimator is 'SVM' else surrogate_estimator

        super(BigDataBRLC, self).__init__(iterations, pos_sign, neg_sign, min_rule_len,
                                          max_rule_len, min_support_pos, min_support_neg, eta, n_chains, alpha,
                                          lambda_, discretize, drop_features)


    def subsample(self, X, y, pos_label=1, neg_label=0):
        """ subsampler to filter the input examples closer to the decision boundary

        Parameters
        -----------
        X : pandas.DataFrame
            input examples representing the training set
        y : pandas.DataFrame
            target labels associated with the training set
        pos_label : int
        neg_label : int

        Returns
        --------
        X_, y_ : pandas.dataframe
        sub-sampled input examples
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise exceptions.DataSetError("Only pandas.DataFrame as input type is currently supported")

        # validate the consistency of the input data
        if not X.shape[0] == y.shape[0]:
            raise exceptions.DataSetError("mismatch in the shape of X and y")

        try:
            self.surrogate_estimator.predict_proba(X[0:1])
        except NotFittedError:
            self.surrogate_estimator.fit(X, y)

        est_prob_scores = pd.DataFrame(self.surrogate_estimator.predict_proba(X))

        # compute the distance from the decision boundary
        distance_from_threshold = est_prob_scores[pos_label].apply(lambda x: np.abs(self.threhold - x))
        pos_label_index = np.where(y == pos_label)[0]
        neg_label_index = np.where(y == neg_label)[0]

        pos_label_dist = distance_from_threshold[pos_label_index]
        neg_label_dist = distance_from_threshold[neg_label_index]

        # sort the neighboring distances from the threshold in the ascending order to select points which
        # are closer to the decision boundary
        sorted_dist_pos_label = pos_label_dist.sort_values()
        sorted_dist_neg_label = neg_label_dist.sort_values()

        # sub-sample the data
        number_of_rows = len(y) * self.sample_percentage
        pos_fraction = len(pos_label_index) / float(len(y))
        neg_fraction = 1 - pos_fraction

        pos_df = pd.DataFrame(X.iloc[sorted_dist_pos_label[:int(number_of_rows * pos_fraction) + 1].index])
        neg_df = pd.DataFrame(X.iloc[sorted_dist_neg_label[:int(number_of_rows * neg_fraction) + 1].index])

        new_X = pd.concat([pos_df, neg_df], axis=0)

        # Randomly shuffle the newly formed data-set
        X_, y_ = shuffle(new_X, y[new_X.index])
        return X_, y_
