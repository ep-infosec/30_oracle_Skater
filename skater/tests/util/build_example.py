import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def build_X(N=1000, dim=3):
    return np.random.normal(0, 10, size=(N, dim))


def build_y_regression(X, b):
    return np.dot(X, b)


def build_y_classifier(X, b, n_classes=2):
    y = build_y_regression(X, b)
    N = len(X)
    n_per_class = int(N / n_classes)
    sorted_inds = y.argsort()

    result = np.zeros(N)

    for i in range(n_classes):
        inds = sorted_inds[i * n_per_class: (i + 1) * n_per_class]
        result[inds] = i

    return np.array(result)


def build_X_y_model(N=1000, dim=3, model_type='classifier', n_classes=2, beta=None):
    if beta is None:
        beta = np.random.normal(0, 1, dim)
    else:
        dim = len(beta)
    X = build_X(N, dim)
    if model_type == 'classifier':
        y = build_y_classifier(X, beta, n_classes=n_classes)
        model = RandomForestClassifier()
        model.fit(X, y)
        return X, y, model
    if model_type == 'regressor':
        y = build_y_regression(X, beta)
        model = RandomForestRegressor()
        model.fit(X, y)
        return X, y, model
