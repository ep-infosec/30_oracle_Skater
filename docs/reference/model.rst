Model Objects
=============

Overview
--------
Skater contains an abstraction for predictive models. Models apis vary by implementation.
The skater Model object manages variations in how models are called, the inputs they expect,
and the outputs they generate, so that inputs, outputs, and calls are standardized to both the
user and to the rest of the code base. Currently the Model object acts as the base class for the
InMemoryModel and DeployedModel class, though this API may change in later versions.

.. code-block:: python
   :linenos:

   from sklearn.datasets import load_breast_cancer
   from sklearn.ensemble import GradientBoostingClassifier

   breast_cancer = load_breast_cancer()
   X = breast_cancer.data
   y = breast_cancer.target

   gb = GradientBoostingClassifier()
   gb.fit(X,y)

   from skater.model import InMemoryModel
   model = InMemoryModel(gb.predict_proba, examples = X)


InMemoryModel
-------------
Models that are callable function are exposed via the InMemoryModel object.

.. automethod:: skater.model.InMemoryModel.__init__



DeployedModel
-------------
Models that are deployed, and therefore callable via http posts are exposed via the
DeployedModel object.

.. automethod:: skater.model.DeployedModel.__init__
