"Subclass of Model for deployed models"
import requests
import numpy as np
from functools import partial

from .base import ModelType


class DeployedModel(ModelType):
    """Model that gets predictions from a deployed model"""
    def __init__(self,
                 uri,
                 input_formatter,
                 output_formatter,
                 request_kwargs={},
                 target_names=None,
                 feature_names=None,
                 unique_values=None,
                 examples=None,
                 model_type=None,
                 probability=None,
                 log_level=30):
        """This model can be called by making http requests
        to the passed in uri.

        Parameters
        ----------
        uri: string
            Where to post requests

        input_formatter: callable
            This function will run on input data before passing
            to requests library. This usually should take array types
            and convert them to JSON.

        output_formatter: callable
            This function will run on outputs before returning
            results to interpretation objects. This usually should take
            request objects and convert them to array types.

        request_kwargs: dict
            any additional request headers that need to be passed, such as api
            keys, content types, etc.

        target_names: array type
            (optional) The names of the target variable/classes. There should be as many
            names as there are outputs per prediction. Defaults to Predicted Value for
            regression and Class 1...n for classification.

        feature_names: array type
            (optional) Names of features the model consumes.

        unique_values: array type
            The set of possible output values. Only use on classifier models that
            return "best guess" predictions, not probability scores, e.g.

            model.predict(fruit1) -> 'apple'
            model.predict(fruit2) -> 'banana'

            ['apple','banana'] are the unique_values of the classifier

        examples:
            optional examples to use to make inferences about the function.

        model_type: None, "classifier", "regressor"
            Indicates which type of model is being used. If left as None, will try to infer based on the
            signature of the output type.

        probability: None, True, False
            If using a classifier, indicates whether probabilities are provided
            (as opposed to indicators/labels).

        log_level: int
            see skater.model.Model for details
        """
        self.uri = uri
        self.input_formatter = input_formatter
        self.output_formatter = output_formatter
        self.request_kwargs = request_kwargs
        super(DeployedModel, self).__init__(examples=examples,
                                            target_names=target_names,
                                            input_formatter=input_formatter,
                                            output_formatter=output_formatter,
                                            log_level=log_level,
                                            unique_values=unique_values,
                                            feature_names=feature_names,
                                            model_type=model_type,
                                            probability=probability,
                                            )


    def _execute(self, data):
        """
        Just use the function itself for predictions
        """
        return requests.post(self.uri, json=data, **self.request_kwargs)


    @staticmethod
    def default_input_wrapper(data, key='input'):
        return {key: data.tolist()}


    @staticmethod
    def default_output_wrapper(response, key='prediction'):
        return np.array(response.json()[key])


    @staticmethod
    def _predict(data, uri, input_formatter, output_formatter, request_kwargs={}, transformer=None):
        """Static prediction function for multiprocessing usecases

        Parameters
        ----------
        data: arraytype

        uri: string
            Where to post requests

        input_formatter: callable
            This function will run on input data before passing
            to requests library. This usually should take array types
            and convert them to JSON.

        output_formatter: callable
            This function will run on outputs before returning
            results to interpretation objects. This usually should take
            request objects and convert them to array types.

        formatter: callable
            function responsible for formatting model outputs as necessary. For instance,
            one hot encoding multiclass outputs.

        predict_fn: callable

        Returns
        -----------
        predictions: arraytype
        """
        query = input_formatter(data)
        response = requests.post(uri, json=query, **request_kwargs)
        results = output_formatter(response)
        if transformer:
            results = transformer(results)
        return results


    def _get_static_predictor(self):

        uri = self.uri
        input_formatter = self.input_formatter
        output_formatter = self.output_formatter
        transformer = self.transformer
        predict_fn = partial(self._predict,
                             uri=uri,
                             input_formatter=input_formatter,
                             output_formatter=output_formatter,
                             transformer=transformer)
        return predict_fn
