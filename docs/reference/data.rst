DataManagers
=======================================

Overview
---------------------------------------
skater contains an abstraction for data, the DataManager. The DataManager
is initialized with some data, and optionally some feature names and indexes. One created, the
DataManager offers a generate_samples method, which includes options for several sampling algorithms.
Any and all updates for handling, accessing, manipulating, saving, and loading data will be
handled by the DataManager to ensure isolation from the rest of the code base.

Currently, skater supports numpy ndarrays and pandas dataframes, with plans on supporting
sparse arrays in future versions.

.. code-block:: python
   :linenos:

   from sklearn.datasets import load_breast_cancer
   from sklearn.ensemble import GradientBoostedClassifier

   breast_cancer = load_breast_cancer()
   X = breast_cancer.data
   y = breast_cancer.target
   features = breast_cancer.feature_names


   from skater.data import DataManager
   data = DataManager(X, feature_names = features)

   data.generate_sample(n, strategy='random-choice')
   data.generate_grid(['CRIM', 'ZN'], grid_resolution=100)



API
---------------------------------------

.. automethod:: skater.data.DataManager.__init__


.. automethod:: skater.data.DataManager.generate_sample


.. automethod:: skater.data.DataManager.generate_grid


.. automethod:: skater.data.DataManager.generate_column_sample
