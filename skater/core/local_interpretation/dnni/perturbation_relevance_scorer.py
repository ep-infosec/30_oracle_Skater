# -*- coding: UTF-8 -*-
import numpy as np

from skater.core.local_interpretation.dnni.initializer import Initializer
from skater.util.logger import build_logger
from skater.util.logger import _INFO


class BasePerturbationMethod(Initializer):
    """
    Base class for perturbation-based relevance/attribution computation

    """

    __name__ = "BasePerturbationMethod"
    logger = build_logger(_INFO, __name__)

    def __init__(self, output_tensor, input_tensor, samples, current_session):
        super(BasePerturbationMethod, self).__init__(output_tensor, input_tensor, samples, current_session)


class Occlusion(BasePerturbationMethod):
    """ Occlusion is a perturbation based inference algorithm. Such forms of algorithm direcly computes the
    relevance/attribution of the input features :math:`(X_{i})` by systematically occluding different
    portions of the image (by removing, masking or altering them), then running a forward pass on the new input to
    produce a new output, and then measuring and monitoring the difference between the original output and new output.
    Perturbation based interpretation helps one to compute direct estimation of the marginal effect of a feature but
    the inference might be computationally expensive depending on the cardinatlity of the feature space.
    The choice of the baseline value while perturbing through the feature space could be set to 0,
    as explained in detail by Zeiler & Fergus, 2014[2].

    References
    ----------
    .. [1] Ancona M, Ceolini E, Oztireli C, Gross M (ICLR, 2018).
    .. Towards better understanding of gradient-based attribution methods for Deep Neural Networks.
    .. [2] Zeiler, M and Fergus, R (Springer, 2014). Visualizing and understanding convolutional networks.
    .. In European conference on computer vision, pp. 818–833.
    .. [3] https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py
    """
    __name__ = "Occlusion"
    logger = build_logger(_INFO, __name__)

    def __init__(self, output_tensor, input_tensor, samples, current_session, **kwargs):
        super(Occlusion, self).__init__(output_tensor, input_tensor, samples, current_session)

        self.input_shape = samples.shape[1:]
        self.replace_value = kwargs['replace_value'] if 'replace_value' in kwargs.keys() else 0
        self.window_size = kwargs['window_size'] if 'window_size' in kwargs.keys() else 1
        self.step = kwargs['step'] if 'step' in kwargs.keys() else 1
        # the input samples are expected to be of the shape,
        # (1, 150, 150, 3) <batch_size, image_width, image_height, no_of_channels>
        self.batch_size = self.samples.shape[0]
        Occlusion.logger.info('Input shape: {}; window_size/step: ({}/{}); replace value: {}; batch size: {}'.
                              format(self.input_shape, self.window_size, self.step, self.replace_value, self.batch_size))


    def _create_masked_input(self, row_value, col_value):
        masked_input = np.array(self.samples)
        # mask the region as set by the window size by replacing the pixel values with the specified value(default:0)
        masked_input[:, row_value:(row_value + self.window_size), col_value:(col_value + self.window_size), :] = self.replace_value
        return masked_input


    def _run(self):
        mask = np.array([self.batch_size, self.window_size, self.window_size, self.samples[0].shape[2]])
        mask.fill(self.replace_value)
        Occlusion.logger.info('Shape of the mask patch: {}'.format(mask.shape))
        relevance_score = np.zeros_like(self.samples, dtype=np.float32)
        # normalizer matrix is set to 1 default; as matrix cell gets used atleast once
        normalizer = np.ones_like(relevance_score)

        # Compute original output
        default_eval = self._session_run(self.output_tensor, self.samples)
        Occlusion.logger.info("shape of the default eval value :{}".format(default_eval.shape))

        count = 1  # to keep track of the number of times a matrix cell is used while perturbing through the feature space
        # Perturb through the feature space by replacing and masking
        for row in range(0, self.samples[0].shape[0] - self.window_size, self.step):
            for col in range(0, self.samples[0].shape[1] - self.window_size, self.step):
                # create masked input while rolling through the input matrix
                new_input = self._create_masked_input(row, col)

                # compute entropy when compared to the original eval value
                delta = default_eval - self._session_run(self.output_tensor, new_input)
                delta_aggregated = np.sum(delta.reshape((self.batch_size, -1)), -1, keepdims=True)
                relevance_score[:, row:(row + self.window_size), col:(col + self.window_size), :] += delta_aggregated

                # keeping track of the number of time a matrix cell is used while perturbing feature space based
                # on window size
                normalizer[:, row:(row + self.window_size), col:(col + self.window_size), :] += (count - 1)

        Occlusion.logger.info("Min/Max normalizer weight: {}/{}".format(np.min(normalizer.shape),
                                                                        np.max(normalizer.shape)))
        relevance_score_norm = relevance_score / normalizer
        Occlusion.logger.info("relevance score matrix shape :{}".format(relevance_score_norm.shape))
        return relevance_score_norm
