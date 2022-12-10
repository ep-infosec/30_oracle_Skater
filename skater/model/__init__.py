"""The modules responsible for wrapping models through a common interface. To enable support
fot classification models with heterogenous output types, and for deployed models that are
accessible through api calls, we need to contruct various patterns of prediction pipelines
and of converting data throughout. Once a prediction pipeline is defined, we use the base Model class
to define how to type inference.
"""

from .local_model import InMemoryModel
from .deployed_model import DeployedModel
from .scorer import Scorer

__all__ = ['InMemoryModel', 'DeployedModel', 'Scorer']
