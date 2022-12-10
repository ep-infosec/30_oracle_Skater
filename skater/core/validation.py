import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics



def compute_validation_curve(estimator, n_folds, x=None, y=None, param_name=None, param_range=None):
    """
    This is a convenient function to compute train and test scores for varying parameter values. Its a simplified form
    of parameter optimization but with one variable. This might come handy once the optimized parameters are
    determined but one wants to investigate and understand the impact of different parameters of the estimator.
    Also, provides training scores for visualization allowing to investigate whether the model is overfitting or
    underfitting.

    Reference:
    ----------
    .. [1] http://scikit-learn.org/stable/modules/learning_curve.html#validation-curves-plotting-scores-to-evaluate-models
    ( sklearn's implementation is tied to the sklearn estimator type hence not allowing comparison with other models
    from other frameworks )
    """
    # TODO: make it scalable
    c_v = StratifiedKFold(n_splits=n_folds)
    param_result_train = []
    param_result_test = []
    for v in param_range:
        cv_result_train = []
        cv_result_test = []
        for train_idx, test_idx in c_v.split(x, y):
            param_dict = {param_name: v}
            estimator.set_params(param_dict)
            estimator.fit(x.iloc[train_idx], y.iloc[train_idx])

            train_scores = estimator.predict_prob(x.iloc[train_idx])
            train_metric = roc_auc_score(y.iloc[train_idx], train_scores[1], pos_label=1)

            test_scores = estimator.predict_proba(x.iloc[test_idx])
            test_metric = roc_auc_score(y.iloc[test_idx], test_scores[1], pos_label=1)
            cv_result_train.append(train_metric[0])
            cv_result_test.append(test_metric[0])
        param_result_train.append(cv_result_train)
        param_result_test.append(cv_result_test)
    return np.array(param_result_train), np.array(param_result_test)


def roc_auc_score(y_true, y_score, pos_label=1):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc, thresholds, fpr, tpr
