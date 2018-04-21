from __future__ import print_function, division

from keras.layers import Activation, BatchNormalization
from keras.layers import Dense, Reshape, Flatten, Dropout, MaxPooling2D, Embedding, LSTM, Average
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model

from neural_networks.nn_implementations.keras_TextNNBase import TextNN_Keras


class TextRNNCNN_Keras(TextNN_Keras):
    def __init__(self, settings: dict):
        super().__init__(settings)

    def build(self):
        # Build and compile the generator
        rnn, rnn_output = self._build_rnn()
        # self.rnn.compile(loss="mse", metrics=['accuracy'], optimizer=optimizer)

        cnn, cnn_output = self._build_cnn()
        # self.cnn.compile(loss="mse", metrics=['accuracy'], optimizer=optimizer)

        combined_output = Average()([rnn_output, cnn_output])
        return Model(inputs=[self.input_x], outputs=[combined_output])

    def _build_rnn(self):
        model = Sequential()
        model.add(
            Embedding(input_dim=self.vocab_size, output_dim=self.settings["embedding_dim"],
                      input_shape=(self.sequence_length,)))
        for i in range(self.settings["lstm_layers"]):
            return_sequences = i < (self.settings["lstm_layers"] - 1)
            model.add(LSTM(self.settings["embedding_dim"], return_sequences=return_sequences))
            model.add(Dropout(self.settings["dropout_keep_prob"]))

        # model.add(Flatten())
        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        result = model(self.input_x)

        return Model(self.input_x, result), result

    def _build_cnn(self):
        model = Sequential()
        model.add(
            Embedding(input_dim=self.vocab_size, output_dim=self.settings["embedding_dim"],
                      input_shape=(self.sequence_length,)))
        model.add(Reshape((self.sequence_length, self.settings["embedding_dim"], 1)))
        for i in range(self.settings["convolution_layers"]):
            model.add(Conv2D(self.settings["num_filters"], 4))
            model.add(Dropout(self.settings["dropout_keep_prob"]))
            model.add(Activation(activation="relu"))
            if self.settings["use_bn"]:
                model.add(BatchNormalization())
        if self.settings["use_max_pooling"]:
            model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        result = model(self.input_x)

        return Model(self.input_x, result), result
