Interpretation Objects
======================

.. _interpretation-overview:

Overview
--------
Interpretation are initialized with a DataManager object, and expose interpretation algorithms as methods. For instance:

.. code-block:: python
   :linenos:

   from skater import Interpretation()
   interpreter = Interpretation()
   interpreter.load_data(data)
   interpreter.feature_importance.feature_importance(model)

Loading Data
------------
Before running interpretation algorithms on a model, the Interpretation object usually needs data, either to learn about
the distribution of the training set or to pass inputs into a prediction function.

When calling Interpretation.load_data, the object creates a DataManager object, which handles the data, keeping track of feature
and observation names, as well as providing various sampling algorithms.

Currently load_data requires a numpy ndarray or pandas DataFrame, though we may add support for additional data structures in the future.
For more details on what the DataManager does, please see the relevant documentation [PROVIDE LINK].

.. currentmodule:: skater
.. automethod:: skater.core.explanations.Interpretation.load_data

.. _global-interpretation:

Global Interpretations
----------------------
A predictive model is a mapping from an input space to an output space. Interpretation algorithms
are divided into those that offer statistics and metrics on regions of the domain, such as the
marginal distribution of a feature, or the joint distribution of the entire training set.
In an ideal world there would exist some representation that would allow a human
to interpret a decision function in any number of dimensions. Given that we generally can only
intuit visualizations of a few dimensions at time, global interpretation algorithms either aggregate
or subset the feature space.

Currently, model agnostic global interpretation algorithms supported by skater include
partial dependence and feature importance.


.. _interpretation-feature-importance:

Feature Importance
~~~~~~~~~~~~~~~~~~
Feature importance is generic term for the degree to which a predictive model relies on a particular
feature. skater feature importance implementation is based on an information theoretic criteria,
measuring the entropy in the change of predictions, given a perturbation of a given feature.
The intuition is that the more a model's decision criteria depend on a feature, the
more we'll see predictions change as a function of perturbing a feature.

Jupyter Notebooks

    1. https://github.com/datascienceinc/Skater/blob/master/examples/ensemble_model.ipynb
    2. https://github.com/datascienceinc/Skater/blob/master/examples/sklearn-classifiers.ipynb
    3. https://github.com/datascienceinc/Skater/blob/master/examples/sklearn_regression_models.ipynb

.. autoclass:: skater.core.global_interpretation.feature_importance.FeatureImportance
   :members:


.. _interpretation-partial-dependence:

Partial Dependence
~~~~~~~~~~~~~~~~~~
Partial Dependence describes the marginal impact of a feature on model prediction, holding
other features in the model constant. The derivative of partial dependence describes the impact of a
feature (analogous to a feature coefficient in a regression model).

Jupyter Notebooks

    1. https://github.com/datascienceinc/Skater/blob/master/examples/ensemble_model.ipynb
    2. https://github.com/datascienceinc/Skater/blob/master/examples/sklearn-classifiers.ipynb
    3. https://github.com/datascienceinc/Skater/blob/master/examples/sklearn_regression_models.ipynb

.. autoclass:: skater.core.global_interpretation.partial_dependence.PartialDependence
   :members:

.. _interpretation-local:


Local Interpretations
---------------------
Local Interpretation could be possibly be achieved in two ways. Firstly, one could possibly approximate the
behavior of a complex predictive model in the vicinity of a single input using a simple interpretable auxiliary or
surrogate model (e.g. Linear Regressor). Secondly, one could use the base estimator to understand the behavior of a
single prediction using intuitive approximate functions based on inputs and outputs.

Local Interpretable Model-Agnostic Explanations(LIME)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LIME is a novel algorithm designed by Riberio Marco, Singh Sameer, Guestrin Carlos to access the behavior of the `any`
base estimator(model) using interpretable surrogate models (e.g. linear classifier/regressor). Such form of
comprehensive evaluation helps in generating explanations which are locally faithful but may not align with the global
behavior.

:Reference:
   Riberio M, Singh S, Guestrin C(2016). Why Should {I} Trust You?": Explaining the Predictions of Any Classifier
   (arXiv:1602.04938v3)

.. autoclass:: skater.core.local_interpretation.lime.lime_tabular.LimeTabularExplainer
   :members:

DNNs: DeepInterpreter
~~~~~~~~~~~~~~~~~~~~~
Helps in interpreting Deep Neural Network Models by computing the relevance/attribution of the output prediction of a
deep network to its input features. The intention is to understand the input-output behavior of the complex network based
on relevant contributing features.

*Define Relevance:* Also known as Attribution or Contribution. Lets define an input
X = :math:`[x1, x2, ... xn] \in R^{n}` to a deep neural network(F) trained for binary
classification (:math:`F(x) \mapsto [0, 1]`). The goal of the relevance/attribution method is to compute
the contribution scores of each input feature :math:`x_{i}` to the output prediction. For e.g. for an image
classification network, if the input :math:`x_{i}` is represented as each pixel of the image, the attribution scores
:math:`(a1, ..., an) \in R^{n}` could inform us which pixels of the image contributed in the selection of the
particular class label.


.. autoclass:: skater.core.local_interpretation.dnni.deep_interpreter.DeepInterpreter
   :members: explain


DNNs: Layerwise Relevance Propagation(e-LRP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: skater.core.local_interpretation.dnni.gradient_relevance_scorer.LRP
   :no-members:
   :no-inherited-members:


DNNs: Integrated Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: skater.core.local_interpretation.dnni.gradient_relevance_scorer.IntegratedGradients
   :no-members:
   :no-inherited-members:


DNNs: Occlusion
~~~~~~~~~~~~~~~
.. autoclass:: skater.core.local_interpretation.dnni.perturbation_relevance_scorer.Occlusion
   :no-members:
   :no-inherited-members:


.. _interpretable-rule-based:

Global And Local Interpretations
--------------------------------

Tree Surrogates (using Decision Trees)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: skater.core.global_interpretation.tree_surrogate.TreeSurrogate
      :members:

Bayesian Rule Lists(BRL)
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: skater.core.global_interpretation.interpretable_models.brlc.BRLC
   :members:
   
.. autoclass:: skater.core.global_interpretation.interpretable_models.bigdatabrlc.BigDataBRLC
   :members:
