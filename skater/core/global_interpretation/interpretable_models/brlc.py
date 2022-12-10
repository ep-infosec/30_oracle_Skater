# coding=utf-8
from skater.util import exceptions

import numbers
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()


class BRLC(object):
    """ :: Experimental :: The implementation is currently experimental and might change in future

    BRLC(Bayesian Rule List Classifier) is a python wrapper for SBRL(Scalable Bayesian Rule list).
    SBRL is a scalable Bayesian Rule List. It's a generative estimator to build hierarchical interpretable
    decision lists. This python wrapper is an extension to the work done by Professor Cynthia Rudin,
    Benjamin Letham, Hongyu Yang, Margo Seltzer and others. For more information check out the reference section below.

    Parameters
    ----------
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
        a hyper-parameter for the expected cardinality of the rules in the optimal rule list.

    n_chains : int (default=10)
        number of chains

    alpha : int (default=1)
        a prior pseudo-count for the positive(alpha1) and negative(alpha0) classes. \
        Default values (1, 1)

    lambda_ : int (default=8)
        a hyper-parameter for the expected length of the rule list.

    discretize : bool (default=True)
        apply discretizer to handle continuous features.

    drop_features : bool (default=False)
        once continuous features are discretized, use this flag to either retain or drop them from the dataframe


    References
    ----------

    .. [1] Letham et.al(2015) Interpretable classifiers using rules and Bayesian analysis:
               Building a better stroke prediction model (https://arxiv.org/abs/1511.01644)

    .. [2] Yang et.al(2016) Scalable Bayesian Rule Lists (https://arxiv.org/abs/1602.08610)

    .. [3] https://github.com/Hongyuy/sbrl-python-wrapper/blob/master/sbrl/C_sbrl.py


    Examples
    --------
    >>> from skater.core.global_interpretation.interpretable_models.brlc import BRLC
    >>> import pandas as pd
    >>> from sklearn.datasets.mldata import fetch_mldata
    >>> input_df = fetch_mldata("diabetes")
    ...
    >>> Xtrain, Xtest, ytrain, ytest = train_test_split(input_df, y, test_size=0.20, random_state=0)
    >>> sbrl_model = BRLC(min_rule_len=1, max_rule_len=10, iterations=10000, n_chains=20, drop_features=True)
    >>> # Train a model, by default discretizer is enabled. So, you wish to exclude features then exclude them using
    >>> # the undiscretize_feature_list parameter
    >>> model = sbrl_model.fit(Xtrain, ytrain, bin_labels="default")
    >>> #print the learned model
    >>> sbrl_inst.print_model()
    >>> features_to_descritize = Xtrain.columns
    >>> Xtrain_filtered = sbrl_model.discretizer(Xtrain, features_to_descritize, labels_for_bin="default")
    >>> predict_scores = sbrl_model.predict_proba(Xtest)
    >>> _, y_hat  = sbrl_model.predict(Xtest)
    >>> # save and reload the model and continue with evaluation
    >>> sbrl_model.save_model("model.pkl")
    >>> sbrl_model.load_model("model.pkl")
    >>> # to access all the learned rules
    >>> sbrl_model.access_learned_rules("all")
    # For a complete example refer to rule_lists_continuous_features.ipynb or rule_lists_titanic_dataset.ipynb notebook
    """
    _estimator_type = "classifier"

    def __init__(self, iterations=30000, pos_sign=1, neg_sign=0, min_rule_len=1,
                 max_rule_len=8, min_support_pos=0.10, min_support_neg=0.10,
                 eta=1.0, n_chains=10, alpha=1, lambda_=10, discretize=True, drop_features=False):

        self.__r_sbrl = importr('sbrl')
        self.model = None
        self.__as_factor = ro.r['as.factor']
        self.__s_apply = ro.r['lapply']
        self.__r_frame = ro.r['data.frame']
        self.model_params = {
            "iters": iterations, "pos_sign": pos_sign, "neg_sign": neg_sign, "rule_minlen": min_rule_len,
            "rule_maxlen": max_rule_len, "minsupport_pos": min_support_pos, "minsupport_neg": min_support_neg,
            "eta": eta, "nchain": n_chains, "lambda": lambda_, "alpha": alpha
        }
        self.__discretize = discretize
        self.__drop_features = drop_features
        self.discretized_features = []
        self.feature_names = []


    def set_params(self, params):
        """ Set model hyper-parameters
        """
        self.model_params[list(params.keys())[0]] = list(params.values())[0]


    def discretizer(self, X, column_list, no_of_quantiles=None, labels_for_bin=None, precision=3):
        """ A discretizer for continuous features

        Parameters
        -----------
        X : pandas.DataFrame
            Dataframe containing continuous features

        column_list : list/tuple

        no_of_quantiles : int or list
            Number of quantiles, e.g. deciles(10), quartiles(4) or as a list of quantiles[0, .25, .5, .75, 1.]
            if 'None' then [0, .25, .5, .75, 1.] is used

        labels_for_bin : labels for the resulting bins

        precision : int
            precision for storing and creating bins


        Returns
        --------
        new_X: pandas.DataFrame
            Contains discretized features


        Examples
        ---------
        >>> sbrl_model = BRLC(min_rule_len=1, max_rule_len=10, iterations=10000, n_chains=20, drop_features=True)
        >>> ...
        >>> features_to_descritize = Xtrain.columns
        >>> Xtrain_discretized = sbrl_model.discretizer(Xtrain, features_to_descritize, labels_for_bin="default")
        >>> predict_scores = sbrl_model.predict_proba(Xtrain_discretized)
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Only pandas.DataFrame as input type is currently supported")
        q_value = [0, .25, .5, .75, 1.] if no_of_quantiles is None else no_of_quantiles
        q_labels = [1, 2, 3, 4] if labels_for_bin is 'default' else labels_for_bin
        new_X = X.copy()
        for column_name in column_list:
            new_clm_name = '{}_q_label'.format(column_name)
            self.discretized_features.append(new_clm_name)
            new_X.loc[:, new_clm_name] = pd.qcut(X[column_name].rank(method='first'), q=q_value,
                                                 labels=q_labels, duplicates='drop', precision=precision)

            # Drop the continuous feature column which has been discritized
            new_X = new_X.drop([column_name], axis=1) if self.__drop_features else new_X
            # explicitly convert the labels column to 'str' type
            new_X = new_X.astype(dtype={'{}_q_label'.format(column_name): "str"})
        return new_X


    def _filter_continuous_features(self, X, column_list=None):
        import collections
        # Sequence is a base class for list and tuple. column_list could be of either type
        if not isinstance(column_list, collections.Sequence):
            raise TypeError("Only list/tuple type supported for specifying column list")
        c_l = X.columns if column_list is None else column_list
        # To check for numeric type, validate again numbers.Number (base class for numeric type )
        # Reference[PEP-3141]: https://www.python.org/dev/peps/pep-3141/
        numeric_type_columns = tuple(filter(lambda c_name: isinstance(X[c_name].iloc[0], numbers.Number), c_l))
        return numeric_type_columns


    # a helper function to filter unwanted features
    filter_to_be_discretize = lambda self, clmn_list, unwanted_list: \
        tuple(filter(lambda c_name: c_name not in unwanted_list, clmn_list))


    def fit(self, X, y_true, n_quantiles=None, bin_labels='default', undiscretize_feature_list=None, precision=3):
        """ Fit the estimator.

        Parameters
        -----------
            X : pandas.DataFrame object, that could be used by the model for training.
                 It must not have a column named 'label'

            y_true : pandas.Series, 1-D array to store ground truth labels

        Returns
        -------
            SBRL model instance: rpy2.robjects.vectors.ListVector


        Examples
        ---------
        >>> from skater.core.global_interpretation.interpretable_models.brlc import BRLC
        >>> sbrl_model = BRLC(min_rule_len=1, max_rule_len=10, iterations=10000, n_chains=20, drop_features=True)
        >>> # Train a model, by default discretizer is enabled. So, you wish to exclude features then exclude them using
        >>> # the undiscretize_feature_list parameter
        >>> model = sbrl_model.fit(Xtrain, ytrain, bin_labels="default")
        """
        if len(np.unique(y_true)) != 2:
            raise Exception("Supports only binary classification right now")

        if not isinstance(X, pd.DataFrame):
            raise exceptions.DataSetError("Only pandas.DataFrame as input type is currently supported")

        # Conditions being managed
        # 1. if 'undiscretize_feature_list' is empty and discretization flag is enabled,
        #    discretize 'all' continuous features
        # 2. if undiscretize_feature_list is not empty and discretization flag is enabled, filter the ones not needed
        #    needed
        for_discretization_clmns = tuple(filter(lambda c_name: c_name not in undiscretize_feature_list, X.columns)) \
            if undiscretize_feature_list is not None else tuple(X.columns)

        data = self.discretizer(X, self._filter_continuous_features(X, for_discretization_clmns),
                                no_of_quantiles=n_quantiles, labels_for_bin=bin_labels, precision=precision) \
            if self.__discretize is True else X

        # record all the feature names
        self.feature_names = data.columns
        data.loc[:, "label"] = y_true
        data_as_r_frame = self.__r_frame(self.__s_apply(data, self.__as_factor))
        self.model = self.__r_sbrl.sbrl(data_as_r_frame, **self.model_params)
        return self.model


    def save_model(self, model_name, compress=True):
        """ Persist the model for future use
        """
        import joblib
        if self.model is not None:
            joblib.dump(self.model, model_name, compress=compress)
        else:
            raise Exception("SBRL model is not fitted yet; no relevant model instance present")


    def load_model(self, serialized_model_name):
        """ Load a serialized model
        """
        import joblib
        try:
            self.model = joblib.load(serialized_model_name)
            # update the BRLC model instance with the the uploaded model
        except (OSError, IOError) as err:
            print("Something is not right with the serialization format. Details {}".format(err))
            raise


    def predict_proba(self, X):
        """ Computes possible class probabilities for the input 'X'

        Parameters
        -----------
            X: pandas.DataFrame object

        Returns
        -------
            pandas.DataFrame of shape (#datapoints, 2), the possible probability of each class for each observation
        """
        if not isinstance(X, pd.DataFrame):
            raise exceptions.DataSetError("Only pandas.DataFrame as input type is currently supported")

        data_as_r_frame = self.__r_frame(self.__s_apply(X, self.__as_factor))
        results = self.__r_sbrl.predict_sbrl(self.model, data_as_r_frame)
        return pandas2ri.ri2py_dataframe(results).T


    def predict(self, X=None, prob_score=None, threshold=0.5, pos_label=1):
        """ Predict the class for input 'X'
        The predicted class is determined by setting a threshold. Adjust threshold to
        balance between sensitivity and specificity

        Parameters
        -----------
        X: pandas.DataFrame
            input examples to be scored
        prob_score: pandas.DataFrame or None (default=None)
            If set to None, `predict_proba` is called before computing the class labels.
            If you have access to probability scores already, use the dataframe of probability scores to compute the
            final class label
        threshold: float (default=0.5)
        pos_label: int (default=1)
            specify how to identify positive label

        Returns
        -------
        y_prob, y_prob['label]: pandas.Series, numpy.ndarray
            Contains the probability score for the input 'X'

        """
        # TODO: Extend it for multi-class classification
        probability_df = self.predict_proba(X) if X is not None and prob_score is None else prob_score
        y_prob = probability_df.loc[:, pos_label]
        y_prob['label'] = np.where(y_prob.values > threshold, 1, 0)
        return y_prob, y_prob['label']


    def print_model(self):
        """ print the decision stumps of the learned estimator
        """
        self.__r_sbrl.print_sbrl(self.model)


    def access_learned_rules(self, rule_indexes="all"):
        """ Access all learned decision rules. This is useful for building and developing intuition

        Parameters
        ----------
        rule_indexes: str (default="all", retrieves all the rules)
            Specify the index of the rules to be retrieved
            index could be set as 'all' or a range could be specified e.g. '(1:3)' will retrieve the rules 1 and 2
        """
        if not isinstance(rule_indexes, str):
            raise TypeError('Expected type string {} provided'.format(type(rule_indexes)))

        # Convert model properties into a readable python dict
        result_dict = dict(zip(self.model.names, map(list, list(self.model))))

        indexes_func = lambda indexes: [int(v) for v in indexes.split(':')]
        # original index starts from 0 while the printed index starts from 1, hence adjust the index
        rules_filter = lambda all_rules, indexes: all_rules['rulenames'][(indexes[0] - 1):(indexes[1] - 1)] \
            if rule_indexes.find(':') > -1 else all_rules['rulenames'][indexes[0] - 1]

        # Enable the ability to access single or multiple sequential model learned decisions
        rules_result = result_dict['rulenames'] if rule_indexes == "all" \
            else rules_filter(result_dict, indexes_func(rule_indexes))
        return rules_result
