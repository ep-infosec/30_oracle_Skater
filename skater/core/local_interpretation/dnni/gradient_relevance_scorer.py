# -*- coding: UTF-8 -*-
from skater.core.local_interpretation.dnni.initializer import Initializer
from skater.util.exceptions import TensorflowUnavailableError
try:
    import tensorflow as tf
except ImportError:
    raise (TensorflowUnavailableError("TensorFlow binaries are not installed"))
import numpy as np
from skater.util.logger import build_logger
from skater.util.logger import _INFO


class BaseGradientMethod(Initializer):
    """
    Base class for gradient-based relevance computation

    Reference
    - https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py
    """

    __name__ = "BaseGradientMethod"
    logger = build_logger(_INFO, __name__)

    def _default_relevance_score(self):
        BaseGradientMethod.logger.debug("Computing default relevance score...")
        return tf.gradients(self.output_tensor, self.input_tensor)


    def _run(self):
        BaseGradientMethod.logger.info("Executing operations ...")
        relevance_scores = self._default_relevance_score()
        results = self._session_run(relevance_scores, self.samples)
        return results[0]


    @classmethod
    def _non_linear_grad(cls, op, grad):
        BaseGradientMethod.logger.debug("Computing gradient with activation type {}".format(op.type))
        return cls._original_grad(op, grad)


class LRP(BaseGradientMethod):
    """ LRP is technique to decompose the prediction(output) of a deep neural networks(DNNs) by computing relevance at
    each layer in a backward pass. Current implementation is computed using backpropagation by applying change rule on
    a modified gradient function. LRP could be implemented in different ways.
    This version implements the epsilon-LRP(Eq (58) as stated in [1] or Eq (2) in [2].
    Epsilon acts as a numerical stabilizer.

    References
    ----------
    .. [1] Bach S, Binder A, Montavon G, Klauschen F, Müller K-R, Samek W (2015)
       On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation.
       PLoS ONE 10(7): e0130140. https://doi.org/10.1371/journal.pone.0130140
    .. [2] Ancona M, Ceolini E, Öztireli C, Gross M:
           Towards better understanding of gradient-based attribution methods for Deep Neural Networks. ICLR, 2018
    """
    __name__ = "LRP"
    _eps = None
    logger = build_logger(_INFO, __name__)

    def __init__(self, output_tensor, input_tensor, samples, session, epsilon=1e-4):
        super(LRP, self).__init__(output_tensor, input_tensor, samples, session)
        assert epsilon > 0.0, 'LRP epsilon must be > 0'
        LRP._eps = epsilon
        LRP.logger.info("Epsilon value: {}".format(LRP._eps))


    def _default_relevance_score(self):
        # computing dot product of the feature wts of the input data and the gradients of the prediction label
        return [g * x for g, x in
                zip(tf.gradients(self.output_tensor, self.input_tensor), [self.input_tensor])]


    @classmethod
    def _non_linear_grad(cls, op, grad):
        LRP.logger.debug("Computing non-linear gradient with activation type {}".format(op.type))
        op_out = op.outputs[0]
        op_in = op.inputs[0]
        stabilizer_epsilon = cls._eps * tf.sign(op_in)
        op_in += stabilizer_epsilon
        return grad * op_out / op_in


class IntegratedGradients(BaseGradientMethod):
    """ Integrated Gradient is a relevance scoring algorithm for Deep network based on final predictions to its input
    features. The algorithm statisfies two fundamental axioms related to relevance/attribution computation,
     1.Sensitivity : For every input and baseline, if the change in one feature causes the prediction to change,
     then the that feature should have non-zero relevance score

     2.Implementation Invariance : Compute relevance(attribution) should be identical for functionally equivalent
     networks.

    References
    ----------
    .. [1] Sundararajan, Mukund, Taly, Ankur, Yan, Qiqi (ICML, 2017).
    .. Axiomatic Attribution for Deep Networks (http://arxiv.org/abs/1703.01365)
    .. [2] Ancona M, Ceolini E, Öztireli C, Gross M (ICLR, 2018).
    .. Towards better understanding of gradient-based attribution methods for Deep Neural Networks.
    .. [3] Taly, Ankur(2017) http://theory.stanford.edu/~ataly/Talks/sri_attribution_talk_jun_2017.pdf
    """
    __name__ = "IntegratedGradients"
    logger = build_logger(_INFO, __name__)

    def __init__(self, output_tensor, input_tensor, samples, session, steps=100, baseline=None):
        super(IntegratedGradients, self).__init__(output_tensor, input_tensor, samples, session)
        self.steps = steps
        self.baseline = self._validate_baseline(baseline)
        # validate baseline, if baseline == None, _validate_baseline assigns the default baseline
        # black image or zero embedding vector for text is used as a default baseline, as suggested in the paper
        # Mukund Sundararajan, Ankir Taly, Qibi Yan. Axiomatic Attribution for Deep Networks(ICML2017)
        self._validate_baseline(self.baseline)


    def _run(self):
        IntegratedGradients.logger.info("Executing operations to compute relevance using Integrated Gradient")
        t_grad = self._default_relevance_score()
        gradient = None
        alpha_list = list(np.linspace(start=1. / self.steps, stop=1.0, num=self.steps))
        for alpha in alpha_list:
            xs_scaled = (self.samples - self.baseline) * alpha
            # compute the gradient for each alpha value
            _scores = self._session_run(t_grad, xs_scaled)
            gradient = _scores if gradient is None else [g + a for g, a in zip(gradient, _scores)]

        results = [(x - b) * (g / self.steps) for g, x, b in zip(gradient, [self.samples], [self.baseline])]
        return results[0]
