# -*- coding: UTF-8 -*-
import warnings
import numpy as np
from tensorflow.python.ops import nn_grad, math_grad

from skater.util.logger import build_logger
from skater.util.logger import _INFO


class Initializer(object):
    """
    """
    __name__ = "Initializer"
    # Currently supported Activation ops
    activation_ops = ['Relu', 'Elu', 'Softplus', 'Tanh', 'Sigmoid']
    _enabled_method_class = None
    _grad_override_checkflag = 0

    logger = build_logger(_INFO, __name__)

    def __init__(self, output_tensor, input_tensor, samples, session):
        self.output_tensor = output_tensor
        self.input_tensor = input_tensor
        self.samples = samples
        self.session = session


    def _session_run(self, output_tensor, samples):
        feed_dict = {}
        feed_dict[self.input_tensor] = samples
        return self.session.run(output_tensor, feed_dict)


    def _validate_baseline(self, baseline):
        if baseline is not None and baseline.shape != ((1,) + self.samples.shape[1:]):
            if baseline.shape == self.samples.shape[1:]:
                baseline = np.expand_dims(baseline, 0)
            else:
                raise RuntimeError('Baseline input shape {} does not match expected input shape {}'
                                   .format(baseline.shape, self.samples.shape[1:]))
        elif baseline is None:
            baseline = np.zeros((1,) + self.samples.shape[1:])
        return baseline


    @classmethod
    def _original_grad(cls, op, grad):
        if op.type not in cls.activation_ops:
            warnings.warn('Selected Activation Ops({}) is currently not supported.'.format(op.type))
        op_name = '_{}Grad'.format(op.type)
        Initializer.logger.debug("Operation name : {}".format(op_name))
        ops_func = getattr(nn_grad, op_name) if hasattr(nn_grad, op_name) else getattr(math_grad, op_name)
        return ops_func(op, grad)
