"""Model subclass for in memory predict functions"""
from functools import partial

from .base import ModelType
from ..util import exceptions


class InMemoryModel(ModelType):
    """
    This model can be called directly from memory
    """

    def __init__(self,
                 prediction_fn,
                 input_formatter=None,
                 output_formatter=None,
                 target_names=None,
                 feature_names=None,
                 unique_values=None,
                 examples=None,
                 model_type=None,
                 probability=None,
                 log_level=30):
        """This model can be called directly from memory

        Parameters
        ----------
        prediction_fn: callable
            function that returns predictions

        input_formatter: callable
            This function will run on input data before passing
            to the prediction_fn. This usually should take your data type
            and convert them to numpy arrays or dataframes.

        output_formatter: callable
            This function will run on input data before passing
            to the prediction_fn. This usually should take your data type
            and convert them to numpy arrays or dataframes.

        target_names: array type
            (optional) names of classes that describe model outputs.

        feature_names: array type
            (optional) Names of features the model consumes.

        unique_values: array type
            The set of possible output values. Only use on classifier models that
            return "best guess" predictions, not probability scores, e.g.

            model.predict(fruit1) -> 'apple'
            model.predict(fruit2) -> 'banana'

            ['apple','banana'] are the unique_values of the classifier

        examples: numpy.array or pandas.dataframe
            optional examples to use to make inferences about the function.

        model_type: None, "classifier", "regressor"
            Indicates which type of model is being used. If left as None, will try to infer based on the
            signature of the output type.

        probability: None, True, False
            If using a classifier, indicates whether probabilities are provided
            (as opposed to indicators/labels).

        log_level: int
            config setting to see model logs. 10 is a good value for seeing debug messages.
            30 is warnings only.


        """

        if not hasattr(prediction_fn, "__call__"):
            raise(exceptions.ModelError("Predict function must be callable"))

        self.prediction_fn = prediction_fn
        super(InMemoryModel, self).__init__(log_level=log_level,
                                            target_names=target_names,
                                            examples=examples,
                                            unique_values=unique_values,
                                            input_formatter=input_formatter,
                                            output_formatter=output_formatter,
                                            feature_names=feature_names,
                                            model_type=model_type,
                                            probability=probability,
                                            )


    def _execute(self, *args, **kwargs):
        """
        Just use the function itself for predictions
        """
        return self.prediction_fn(*args, **kwargs)


    @staticmethod
    def _predict(data, predict_fn, input_formatter, output_formatter, transformer):
        """Static prediction function for multiprocessing usecases

        Parameters
        ----------
        data: arraytype

        formatter: callable
            function responsible for formatting model outputs as necessary. For instance,
            one hot encoding multiclass outputs.

        predict_fn: callable

        Returns
        -----------
        predictions: arraytype
        """
        results = output_formatter(predict_fn(input_formatter(data)))
        if transformer:
            return transformer(results)
        else:
            return results


    def _get_static_predictor(self):

        predict_fn = partial(self._predict,
                             transformer=self.transformer,
                             predict_fn=self.prediction_fn,
                             input_formatter=self.input_formatter,
                             output_formatter=self.output_formatter,
                             )
        return predict_fn
