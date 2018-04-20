from __future__ import print_function, division

from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.models import Sequential, Model

from neural_networks.nn_implementations.keras_TextNNBase import TextNN_Keras


class TextRNN_Keras(TextNN_Keras):
    def __init__(self, settings: dict):
        super().__init__(settings)

    def build(self):
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

        return Model(self.input_x, result)
