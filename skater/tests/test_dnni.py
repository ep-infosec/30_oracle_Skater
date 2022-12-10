import unittest

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model, model_from_yaml
from keras import backend as K

from skater.core.local_interpretation.dnni.deep_interpreter import DeepInterpreter


class TestDNNI(unittest.TestCase):

    # The below architecture maps to the pre_trained_models/mnist_cnn/model_mnist_cnn_epoch_3*
    # @classmethod
    # def _build_mnist_CNN_model(cls):
    #     from keras.layers import Dense, Dropout, Flatten, Activation
    #     from keras.layers import Conv2D, MaxPooling2D
    #
    #     Build and train a network.
    #     model = Sequential()
    #     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=cls.input_shape))
    #     model.add(Conv2D(64, (3, 3), activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.25))
    #     model.add(Flatten())
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(cls.num_classes))
    #     model.add(Activation('softmax'))
    #
    #     model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
    #                   metrics=['accuracy'])
    #     model.fit(cls.x_train, cls.y_train, batch_size=cls.batch_size, epochs=cls.epochs,
    #               verbose=1, validation_data=(cls.x_test, cls.y_test))
    #     return model


    @classmethod
    def _load(cls, model_name):
        f_h = open('{}.yaml'.format(model_name), 'r')
        persisted_yaml_model = f_h.read()
        f_h.close()
        loaded_model = model_from_yaml(persisted_yaml_model)
        # load weights into retrieved model instance
        loaded_model.load_weights("{}.h5".format(model_name))
        return loaded_model


    @classmethod
    def setUpClass(cls):
        # MNIST dataset used for building pre_trained_models/mnist_cnn/model_mnist_cnn_epoch_3
        cls.batch_size = 128
        cls.num_classes = 10
        cls.epochs = 2
        # input image dimensions
        cls.img_rows, cls.img_cols = 28, 28
        # shuffled and split between train and test sets
        (cls.x_train, cls.y_train), (cls.x_test, cls.y_test) = mnist.load_data()
        if K.image_data_format() == 'channels_first':
            cls.x_train = cls.x_train.reshape(cls.x_train.shape[0], 1, cls.img_rows, cls.img_cols)
            cls.x_test = cls.x_test.reshape(cls.x_test.shape[0], 1, cls.img_rows, cls.img_cols)
            cls.input_shape = (1, cls.img_rows, cls.img_cols)
        else:
            cls.x_train = cls.x_train.reshape(cls.x_train.shape[0], cls.img_rows, cls.img_cols, 1)
            cls.x_test = cls.x_test.reshape(cls.x_test.shape[0], cls.img_rows, cls.img_cols, 1)
            cls.input_shape = (cls.img_rows, cls.img_cols, 1)

        cls.x_train = cls.x_train.astype('float32')
        cls.x_test = cls.x_test.astype('float32')
        cls.x_train /= 255
        cls.x_test /= 255
        cls.x_train = (cls.x_train - 0.5) * 2
        cls.x_test = (cls.x_test - 0.5) * 2

        # convert class vectors to binary class matrices
        cls.y_train = keras.utils.to_categorical(cls.y_train, cls.num_classes)
        cls.y_test = keras.utils.to_categorical(cls.y_test, cls.num_classes)


    def test_deep_interpreter_cnn(self):
        K.set_learning_phase(0)
        with DeepInterpreter(session=tf.compat.v1.keras.backend.get_session()) as di:
            # 1. Load the persisted model
            # 2. Retrieve the input tensor from the loaded model

            retrieved_model = self._load('skater/tests/pre_trained_models/mnist_cnn/model_mnist_cnn_epoch_3')
            input_tensor = retrieved_model.layers[0].input
            output_tensor = retrieved_model.layers[-2].output

            # 3. We will using the last dense layer(pre-softmax) as the output layer
            # 4. Instantiate a model with the new input and output tensor
            new_model = Model(inputs=input_tensor, outputs=output_tensor)
            target_tensor = new_model(input_tensor)
            xs = self.x_test[0:2]
            ys = self.y_test[0:2]

            relevance_scores_elrp = di.explain('elrp', target_tensor * ys, input_tensor, xs, use_case='image')
            relevance_scores_ig = di.explain('ig', target_tensor * ys, input_tensor, xs, use_case='image')
            # test for optional arguments related to window_size and step size for 'occlusion'
            optional_args = {"window_size": 2, "step": 2}
            relevance_scores_occ = di.explain('occlusion', target_tensor * ys, input_tensor, xs,
                                              use_case='image', **optional_args)
        self.assertEquals(relevance_scores_elrp.shape, (2, 28, 28, 1))
        self.assertEquals(relevance_scores_ig.shape, (2, 28, 28, 1))
        self.assertEquals(relevance_scores_occ.shape, (2, 28, 28, 1))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDNNI)
    unittest.TextTestRunner(verbosity=2).run(suite)
