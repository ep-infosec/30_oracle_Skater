"""Static types class and methods for inferring types."""
import numpy as np
import six


class ModelTypes(object):
    """Stores values for model types and keywords"""
    regressor = 'regressor'
    classifier = 'classifier'
    unknown = 'unknown'

    _valid_ = [regressor, classifier]


class OutputTypes(object):
    """Stores values for output types, and keywords"""
    float = 'float'
    int = 'int'
    string = 'string'
    iterable = 'iterable'
    numeric = 'numeric'
    unknown = 'unknown'

    _valid_ = [float, int, string, iterable, numeric]


class DataTypes(object):

    @staticmethod
    def is_numeric(thing):
        try:
            float(thing)
            return True
        except ValueError:
            return False
        except TypeError:
            return False


    @staticmethod
    def is_string(thing):
        return isinstance(thing, (six.text_type, six.binary_type))


    @staticmethod
    def is_dtype_numeric(dtype):
        assert isinstance(dtype, np.dtype), "expect numpy.dtype, got {}".format(type(dtype))
        return np.issubdtype(dtype, np.number)

    @staticmethod
    def return_data_type(thing):
        """Returns an output type given a variable"""
        if isinstance(thing, (six.text_type, six.binary_type)):
            return StaticTypes.output_types.string
        elif isinstance(thing, int):
            return StaticTypes.output_types.int
        elif isinstance(thing, float):
            return StaticTypes.output_types.float
        elif DataTypes.is_numeric(thing):
            return StaticTypes.output_types.numeric
        elif hasattr(thing, "__iter__"):
            return StaticTypes.output_types.iterable
        else:
            return StaticTypes.unknown


class ScorerTypes(object):
    """Stores values for scorer types"""
    increasing = 'increasing'
    decreasing = 'decreasing'


class StaticTypes(object):
    """Stores values for model types, output types, and keywords"""
    model_types = ModelTypes
    output_types = OutputTypes
    data_types = DataTypes
    scorer_types = ScorerTypes
    unknown = 'unknown'
    not_applicable = 'not applicable'
