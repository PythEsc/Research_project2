from __future__ import print_function, division

from keras.layers import Activation, Dropout
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D, Embedding, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model

from neural_networks.nn_implementations.keras_TextNNBase import TextNN_Keras


class TextCNN_Keras(TextNN_Keras):

    def __init__(self, settings: dict):
        super().__init__(settings)

    def build(self):
        model = Sequential()
        model.add(
            Embedding(input_dim=self.vocab_size, output_dim=self.settings["embedding_dim"],
                      input_shape=(self.sequence_length,)))
        model.add(Reshape((self.sequence_length, self.settings["embedding_dim"], 1)))
        for i in range(self.settings["convolution_layers"]):
            model.add(Conv2D(self.settings["num_filters"], 4))
            model.add(Activation(activation="relu"))
            if self.settings["use_bn"]:
                model.add(BatchNormalization())
            model.add(Dropout(self.settings["dropout_keep_prob"]))
        if self.settings["use_max_pooling"]:
            model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        result = model(self.input_x)

        return Model(self.input_x, result)
