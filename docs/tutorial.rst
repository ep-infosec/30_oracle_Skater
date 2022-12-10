********
Tutorial
********

A. Post-hoc Model Interpretation
################################


Creating an interpretation object
---------------------------------

The general workflow within the skater package is to create an interpretation, create a model, and run interpretation algorithms.

An Interpretation consumes a dataset, and optionally some meta data like feature names and row ids.
Internally, the Interpretation will generate a DataManager to handle data requests and sampling.

.. code-block:: python

   from sklearn.datasets import load_boston
   boston = load_boston()
   X, y, features = boston.data, boston.target, boston.feature_names

   from skater import Interpretation
   interpreter = Interpretation(X, feature_names=features)

To begin using the Interpretation to explain models, we need to create a skater Model.


Local Models(InMemoryModel)
---------------------------
To create a skater model based on a local function or method, pass in the predict function to an InMemoryModel. A user can optionally pass data samples to the examples keyword argument. This is only used to infer output types and formats. 
Out of the box, skater allows models return numpy arrays and pandas dataframes. If you would like support for additional formats, please let us know: https://github.com/datascienceinc/model-interpretation/issues/117

.. code-block:: python

   from sklearn.ensemble import GradientBoostedRegressor
   gb = GradientBoostedRegressor()
   gb.fit(X, y)

   from skater.model import InMemoryModel
   model = InMemoryModel(gb.predict, examples = X[:10])

If your model requires or returns different data structures, you can instead create a function that converts outputs to an appropriate data structure.

.. code-block:: python

   def predict_as_dataframe(x):
       return pd.DataFrame(gb.predict(x))

   from skater.model import InMemoryModel
   model = InMemoryModel(predict_as_dataframe, examples = X[:10])

Operationalized Model(DeployedModel)
------------------------------------
If your model is accessible through an api, use a DeployedModel, which wraps the requests library. DeployedModels require two functions,
an input formatter and an output formatter, which speak to the requests library for posting and parsing.

The input formatter takes a pandas DataFrame or a numpy ndarray, and returns an object (such as a dict) that can be converted to json
to be posted. The output formatter takes a requests.response as an input and returns a numpy ndarray or pandas DataFrame:

.. code-block:: python

   from skater.model import DeployedModel
   import numpy as np

   def input_formatter(x): return {'data': list(x)}
   def output_formatter(response): return np.array(response.json()['output'])
   uri = "https://yourorg.com/model/endpoint"
   model = DeployedModel(uri, input_formatter, output_formatter, examples = X[:10])


If your api requires additional configuration like cookies, use request_kwargs:

.. code-block:: python

   from skater.model import DeployedModel
   import numpy as np

   req_kwargs = {'cookies': {'cookie-name':'cookie'}}
   model = DeployedModel(uri, input_formatter, output_formatter, examples = X[:10], request_kwargs=req_kwargs)


Model Input/Output Data Types
-----------------------------
Skater natively supports models that accept numpy arrays and pandas dataframes as inputs.
If your model requires a different input type, such as the case of a model API
requiring JSON, or an H20 model requiring a H20Frame, then you'll need to include
an input formatter function to the Skater Model object, for example:


::

   def numpy_to_json(numpy_array):
      return [{'data':x} for x in numpy_array]

    skater_model = InMemoryModel(model.predict, input_formatter = numpy_to_json)


Likewise, Skater natively supports models that return numpy arrays or pandas dataframes.
If your model returns another data structure, you'll need to define an output_formatter
that takes your model's return type, and returns a numpy array or pandas dataframe.

Model Types
-----------
Skater supports regressions, classifiers with or without probability scores.

Skater expects that regression models run on 'n' examples will return numerical arrays
of shape (n, ) or (n, 1), such as the following regression output run on 3 examples:

::

   np.array([1.2, -2.2, 3.1])


Skater expects that classification models with probability scores of k classes run
on n examples will return numerical arrays of shape (n, k), where elements are
between 0 and 1 and rows sum to 1, such as the following classifier output run
on 4 examples with 3 classes:

::

   np.array([.0, .32, .68],
            [.1, .2,  .7],
            [.5, .5,  .0],
            [.8, .1,  .1])

Skater expects that classification models without probability scores of k classes run
on n examples will return arrays of shape (n, ) or (n, 1), such as the following
classifier run on 3 examples with 2 classes:

::

   np.array(['apple','banana','banana'])

or

::

   np.array([0, 1, 1])

Note that in this last case of classifiers that do not provide probabilities for
all classes, there is no implicit definition of types of classes the model can
predict. Therefore, these models require `unique_values` keyword argument when
initializing a Skater model, which defines the unique classes that a model might
return,

such as:

::

    unique_classes = [0, 1]
    skater_model = InMemoryModel(classifier.predict, unique_classes=unique_classes)

or

::

    unique_classes = ['apple','banana']
    skater_model = InMemoryModel(classifier.predict, unique_classes=unique_classes)


With an Interpretation and a Model, one can access golabl interpretation algorithms.

.. code-block:: python

   interpreter.feature_importance.feature_importance(skater_model)

   interpreter.partial_dependence.plot_partial_dependence([features[0], features[1]], skater_model)
   
   
For Local Interpretation, one can access LIME as,

.. code-block:: python

   from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
   LimeTabularExplainer(regressor_X, feature_names=regressor_data.feature_names,
   mode="regression").explain_instance(regressor_X[0], annotated_model)
   

B. Natively interpretable models(Rule Based Models/Transparent Design)
######################################################################

For Global and Local Interpretation(Transparent Models), Skater support Rule based models using Bayesian Rule Lists,

.. code-block:: python

   from skater.core.global_interpretation.interpretable_models.brlc import BRLC
   sbrl_model = BRLC(min_rule_len=1, max_rule_len=10, iterations=10000, n_chains=20, drop_features=True)


For details on the interpretation algorithms currently available, please see the documentation for:

- :ref:`global-interpretation`
- :ref:`interpretation-feature-importance`
- :ref:`interpretation-partial-dependence`
- :ref:`interpretation-local`
- :ref:`interpretable-rule-based`
