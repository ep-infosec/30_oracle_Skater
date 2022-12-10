"Specialized Exceptions for skater"


def exception_factory(exception_name, base_exception=Exception, attributes=None):
    attribute_dict = {
        "__init__": base_exception.__init__
    }
    if isinstance(attributes, dict):
        attributes.update(attributes)
    return type(
        exception_name,
        (base_exception, ),
        attribute_dict
    )


DataSetNotLoadedError = exception_factory('DataSetNotLoadedError')

PartialDependenceError = exception_factory('PartialDependenceError')

FeatureImportanceError = exception_factory('FeatureImportanceError')

DataSetError = exception_factory('DataSetError')

ModelError = exception_factory("ModelError")

TooManyFeaturesError = exception_factory('TooManyFeaturesError',
                                         base_exception=PartialDependenceError)

DuplicateFeaturesError = exception_factory('DuplicateFeaturesError',
                                           base_exception=PartialDependenceError)

EmptyFeatureListError = exception_factory('EmptyFeatureListError',
                                          base_exception=PartialDependenceError)

MalformedGridError = exception_factory("MalformedGridError",
                                       base_exception=PartialDependenceError)

MalformedGridRangeError = exception_factory("MalformedGridRangeError",
                                            base_exception=PartialDependenceError)

MatplotlibUnavailableError = exception_factory('MatplotlibUnavailableError', base_exception=ImportError)

PlotlyUnavailableError = exception_factory('PlotlyUnavailableError', base_exception=ImportError)

TensorflowUnavailableError = exception_factory('TensorflowUnavailableError', base_exception=ImportError)

KerasUnavailableError = exception_factory('KerasUnavailableError', base_exception=ImportError)

MatplotlibDisplayError = exception_factory('MatplotlibDisplayError', base_exception=RuntimeError)
