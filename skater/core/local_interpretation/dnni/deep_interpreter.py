# -*- coding: UTF-8 -*-
from tensorflow.python.framework import ops
import tensorflow as tf

from collections import OrderedDict
import warnings

from skater.core.local_interpretation.dnni.gradient_relevance_scorer import BaseGradientMethod
from skater.core.local_interpretation.dnni.gradient_relevance_scorer import LRP
from skater.core.local_interpretation.dnni.gradient_relevance_scorer import IntegratedGradients
from skater.core.local_interpretation.dnni.perturbation_relevance_scorer import Occlusion
from skater.core.local_interpretation.dnni.initializer import Initializer

from skater.util.logger import build_logger
from skater.util.logger import _WARNING
from skater.util.logger import _INFO

logger = build_logger(_INFO, __name__)


@ops.RegisterGradient("DeepInterpretGrad")
def deep_interpreter_grad(op, grad):
    logger.debug("Computing gradient using DeepInterpretGrad")
    Initializer._grad_override_checkflag = 1
    if Initializer._enabled_method_class is not None \
            and issubclass(Initializer._enabled_method_class, BaseGradientMethod):
        logger.debug("Computing gradient using DeepInterpretGrad: {}".
                     format(Initializer._enabled_method_class._non_linear_grad))
        return Initializer._enabled_method_class._non_linear_grad(op, grad)
    else:
        logger.debug("Computing gradient using DeepInterpretGrad: {}".format(Initializer._original_grad))
        return Initializer._original_grad(op, grad)


class DeepInterpreter(object):
    """ :: Experimental :: The implementation is currently experimental and might change in future
    Interpreter for inferring Deep Learning Models. Given a trained NN model and an input vector X, DeepInterpreter
    is responsible for providing relevance scores w.r.t a target class to analyze most contributing features driving
    an estimator's decision for or against the respective class

    Framework supported: Tensorflow(>=1.4.0) and Keras(>=2.0.8)

    Parameters
    ----------
    graph : tensorflow.Graph instance
    session : tensorflow.Session to execute the graph(default session: tf.compat.v1.get_default_session)
    log_level : int (default: _WARNING)
        The log_level could be adjusted to other values as well. Check here `./skater/util/logger.py`

    References
    ----------
    .. [1] Ancona M, Ceolini E, Oztireli C, Gross M (ICLR, 2018).
           Towards better understanding of gradient-based attribution methods for Deep Neural Networks.
           https://arxiv.org/abs/1711.06104
           (https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py)
    """
    __name__ = "DeepInterpreter"

    def __init__(self, graph=None, session=tf.compat.v1.get_default_session, log_level=_WARNING):
        self.logger = build_logger(log_level, __name__)
        self.relevance_type = None
        self.use_case_str = None
        self.batch_size = None
        self.session = session
        if self.session is None:
            raise RuntimeError('Relevant session not retrieved')
        else:
            self.logger.info("Current session: {}".format(session.__dict__))
        self.graph = session.graph if graph is None else graph
        # request for the default graph
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self._get_gradient_override_map())
        self.context_on = False
        self.__supported_relevance_type_dict = OrderedDict({
            'elrp': {'use_case_type': ['image'], 'method': LRP},
            'ig': {'use_case_type': ['image', 'txt'], 'method': IntegratedGradients},
            'occlusion': {'use_case_type': ['image'], 'method': Occlusion}
        })


    def __enter__(self):
        # Magic method for managing context with the usage of 'with' statement ( context-guard ).
        # This helps in managing an active keras/tensorflow session for resolving variable scope
        # reference: http://effbot.org/zone/python-with-statement.htm
        # Override gradient of all ops created in context
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.context_on = True
        return self


    def __exit__(self, type, value, traceback):
        # Exit a `with` block
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)
        self.context_on = False


    @staticmethod
    def _get_gradient_override_map():
        return dict((ops_item, 'DeepInterpretGrad') for ops_item in Initializer.activation_ops)


    def _validate_relevance_type(self, type_name, use_case_str):
        supported_type = self.__supported_relevance_type_dict[type_name] \
            if type_name in self.__supported_relevance_type_dict.keys() else None
        if supported_type is None:
            raise RuntimeError('Method type not found in {}'.format(list(self.__supported_relevance_type_dict.keys())))
        else:
            use_case_list = self.__supported_relevance_type_dict[type_name]['use_case_type']
            if use_case_str not in use_case_list:
                raise RuntimeError('Method to use-case map is not supported {}')
            else:
                return self.__supported_relevance_type_dict[type_name]['method']


    def explain(self, relevance_type, output_tensor, input_tensor, samples, use_case=None, **kwargs):
        """ Helps in computing the relevance scores for DNNs to understand the input and output behavior of the network.

        Parameters
        ----------
        relevance_type: str
            Currently, relevance score could be computed using e-LRP('elrp') or Integrated Gradient('ig'). Other
            algorithms are under development.

             - epsilon-LRP('eLRP'):
               Is recommended with Activation ops ('ReLU' and 'Tanh'). Current implementation of
               LRP works only for images and makes use of epsilon(default: 0.0001) as a stabilizer.

             - Integrated Gradient('ig'):
               Is recommended with Activation ops ('Relu', 'Elu', 'Softplus', 'Tanh', 'Sigmoid').
               It works for images and text. Optional parameters include steps(default: 100) and
               baseline(default: {'image': 'a black image'}; {'txt': zero input embedding vector})
               Gradient is computed by varying the input from the baseline(x') to the provided input(x). x, x'
               are element of R with n dimension ---> [0,1]
        output_tensor: tensorflow.python.framework.ops.Tensor
            Specify the output layer to start from
        input_tensor: tensorflow.python.framework.ops.Tensor
            Specify the input layer to reach to
        samples: numpy.array
            Batch of input for which explanations are desired.
            Note: The first dimension of the array specifies the batch size. For e.g.,
                  - for an image input of batch size 2: (2, 150, 150, 3) <batch_size, image_width, image_height, no_of_channels>
                  - for a text input of batch size 1: (1, 80) <batch_size, embedding_dimensions>
        use_case: str
            Options: 'image' or 'txt
        kwargs: optional

        Returns
        -------
        result: numpy.ndarray
            Computed relevance(contribution) score for the given input

        References
        ----------
        .. [1] Bach S, Binder A, Montavon G, Klauschen F, Müller K-R, Samek W (2015)
           On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation.
           PLoS ONE 10(7): e0130140. https://doi.org/10.1371/journal.pone.0130140
        .. [2] Sundararajan, M, Taly, A, Yan, Q (ICML, 2017).
           Axiomatic Attribution for Deep Networks. http://arxiv.org/abs/1703.01365
        .. [3] Ancona M, Ceolini E, Öztireli C, Gross M (ICLR, 2018).
           Towards better understanding of gradient-based attribution methods for Deep Neural Networks.
           https://arxiv.org/abs/1711.06104

        Examples
        --------
        >>> from skater.core.local_interpretation.dnni.deep_interpreter import DeepInterpreter
        >>> ...
        >>> import keras
        >>> from keras.datasets import mnist
        >>>from keras.models import Sequential, Model, load_model, model_from_yaml
        >>> from keras.layers import Dense, Dropout, Flatten, Activation
        >>> from keras.layers import Conv2D, MaxPooling2D
        >>> from keras import backend as K
        >>> import tensorflow as tf
        >>> import matplotlib.pyplot as plt
        >>> sess = tf.Session()
        >>> K.set_session(sess)
        >>> ... # Load dataset
        >>> # A simple network for MNIST data-set using Keras
        >>> model = Sequential()
        >>> model.add(Conv2D(32, kernel_size=(3, 3),
        >>>         activation='relu',
        >>>         input_shape=input_shape))
        >>> model.add(Conv2D(64, (3, 3), activation='relu'))
        >>> model.add(MaxPooling2D(pool_size=(2, 2)))
        >>> model.add(Dropout(0.25))
        >>> model.add(Flatten())
        >>> model.add(Dense(128, activation='relu'))
        >>> model.add(Dropout(0.5))
        >>> model.add(Dense(num_classes))
        >>> model.add(Activation('softmax'))
        >>> ... # Compile and train the model
        >>> K.set_learning_phase(0)
        >>> with DeepInterpreter(session=K.get_session()) as di:
        >>>   # 1. Load the persisted model
        >>>   # 2. Retrieve the input tensor from the loaded model
        >>>   yaml_file = open('model_sample.yaml', 'r')
        >>>   loaded_model_yaml = yaml_file.read()
        >>>   yaml_file.close()
        >>>   loaded_model = model_from_yaml(loaded_model_yaml)
        >>>   # load weights into new model
        >>>   loaded_model.load_weights("model_mnist_cnn_3.h5")
        >>>   print("Loaded model from disk")
        >>>   input_tensor = loaded_model.layers[0].input
        >>>   output_tensor = loaded_model.layers[-2].output

        >>>    # 3. We will using the last dense layer(pre-softmax) as the output layer
        >>>    # 4. Instantiate a model with the new input and output tensor
        >>>    new_model = Model(inputs=input_tensor, outputs=output_tensor)
        >>>    target_tensor = new_model(input_tensor)
        >>>    xs = input_x
        >>>    ys = input_y
        >>>    print("X shape: {}".format(xs.shape))
        >>>    print("Y shape: {}".format(ys.shape))
        >>>    # Original Predictions
        >>>    print(loaded_model.predict_classes(xs))
        >>>    relevance_scores = di.explain('elrp', output_tensor=target_tensor * ys, input_tensor=input_tensor,
        >>>                                                                                samples=xs, use_case='image')
        """
        if not self.context_on:
            raise RuntimeError('explain can be invoked only within a DeepInterpreter context.')

        self.logger.info("all supported relevancy scorers {}".format(self.__supported_relevance_type_dict))

        # Validate if the specified relevance type is supported.
        self.relevance_type = relevance_type
        self.use_case_str = use_case
        relevance_type_class = self._validate_relevance_type(self.relevance_type, self.use_case_str)
        if relevance_type_class is None:
            raise RuntimeError('Method type not found in {}'.format(list(self.__supported_relevance_type_dict.keys())))
        self.logger.info('DeepInterpreter: executing relevance type class {}'.format(relevance_type_class))

        Initializer._grad_override_checkflag = 0
        Initializer._enabled_method_class = relevance_type_class

        method = Initializer._enabled_method_class(output_tensor, input_tensor, samples, self.session, **kwargs)
        self.logger.info('DeepInterpreter: executing method {}'.format(method))

        result = method._run()
        if issubclass(Initializer._enabled_method_class, BaseGradientMethod) and Initializer._grad_override_checkflag == 0:
            warnings.warn('Results may not reliable: As default gradient seems to have been used. '
                          'or you might have forgotten to create the graph within the DeepInterpreter context. '
                          'Be careful...')

        Initializer._enabled_method_class = None
        Initializer._grad_override_checkflag = 0
        return result
