from __future__ import print_function, division

import numpy as np
import sklearn.metrics
from keras import losses, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, Embedding, LSTM, Average
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.contrib import learn

from importer.database.mongodb import MongodbStorage
from neural_networks.util.data_helpers import get_training_set, clean_text
from pre_trained_embeddings.word_embeddings import WordEmbeddings


class TextRNNCNN_Keras():
    def __init__(self, settings: dict):
        self.settings = settings
        self.vocab_processor = None
        self.checkpoint_path = settings["checkpoint_path"]
        self.lstm_layers = settings["lstm_layers"]

        # Read in the data
        self.x, self.y = self.read_data()
        # Get the dimensions for the network by reading the data
        self.sequence_length = self.x.shape[1]
        self.num_classes = self.y.shape[1]
        self.vocab_size = len(self.vocab_processor.vocabulary_)

        # Initialize the network
        # Start with creating placeholder variables
        self.input_x = Input((self.sequence_length,), name="input_x")
        self.input_y = Input((self.num_classes,), name="input_y")

        optimizer = Adam(lr=self.settings["learning_rate"], beta_1=0.5)

        # Build and compile the generator
        self.rnn, self.rnn_output = self.build_rnn()
        # self.rnn.compile(loss="mse", metrics=['accuracy'], optimizer=optimizer)

        self.cnn, self.cnn_output = self.build_cnn()
        # self.cnn.compile(loss="mse", metrics=['accuracy'], optimizer=optimizer)

        self.combined_output = Average()([self.rnn_output, self.cnn_output])
        self.combined = Model(inputs=[self.input_x], outputs=[self.combined_output])
        self.combined.compile(loss=losses.categorical_crossentropy,
                              metrics=[metrics.categorical_crossentropy, metrics.categorical_accuracy],
                              optimizer=optimizer)

    def build_rnn(self):
        model = Sequential()
        model.add(
            Embedding(input_dim=self.vocab_size, output_dim=self.settings["embedding_dim"],
                      input_shape=(self.sequence_length,)))
        for i in range(self.lstm_layers):
            return_sequences = i < (self.lstm_layers - 1)
            model.add(LSTM(self.settings["embedding_dim"], return_sequences=return_sequences))

        # model.add(Flatten())
        model.add(Dropout(self.settings["dropout_keep_prob"]))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        result = model(self.input_x)

        return Model(self.input_x, result), result

    def build_cnn(self):
        model = Sequential()
        model.add(
            Embedding(input_dim=self.vocab_size, output_dim=self.settings["embedding_dim"],
                      input_shape=(self.sequence_length,)))
        model.add(Reshape((self.sequence_length, self.settings["embedding_dim"], 1)))
        for i in range(self.settings["convolution_layers"]):
            model.add(Conv2D(40, 4))
            model.add(Activation(activation="relu"))
            if self.settings["use_bn"]:
                model.add(BatchNormalization())
            if self.settings["use_max_pooling"]:
                model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dropout(self.settings["dropout_keep_prob"]))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        result = model(self.input_x)

        return Model(self.input_x, result), result

    def read_data(self):
        # Data Preparation
        # ==================================================

        # Load data
        print("Loading data...")

        db = MongodbStorage()
        x_text, y = get_training_set(db)
        x_text = clean_text(x_text)

        # Load pre-trained word embedding
        print('\nLoading pre-trained word embedding...')
        embedding_dim = 50
        we = WordEmbeddings(embedding_dim)

        # Create feature matrix using vocab from trained WordEmbeddings
        print('\nCreating feature matrix...')
        document_legths = [len(x.split(" ")) for x in x_text]
        max_document_length = max(document_legths)
        min_document_length = min(document_legths)
        mean_document_length = int(sum(document_legths) / len(document_legths))
        print('Max document length is {}.'.format(max_document_length))
        print('Mean document length is {}.'.format(mean_document_length))
        print('Min document length is {}.'.format(min_document_length))
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(mean_document_length,
                                                                       vocabulary=we.categorical_vocab)
        x = np.array(list(self.vocab_processor.transform(x_text)))
        y = np.array(y)
        words_not_in_we = len(self.vocab_processor.vocabulary_) - len(we.vocab)
        if words_not_in_we > 0:
            print("Words not in pre-trained vocab: {:d}".format(words_not_in_we))

        return x, y

    def train(self):

        # Generate batches
        self.combined.fit(self.x, self.y, batch_size=self.settings["batch_size"], epochs=self.settings["num_epochs"],
                          validation_split=self.settings["dev_sample_percentage"], shuffle=True, verbose=2,
                          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.005, patience=8),
                                     ModelCheckpoint(filepath=self.settings["checkpoint_path"] + "cnn_best.model",
                                                     save_best_only=True),
                                     ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=0.0005)])

    def predict(self, content: list) -> list:
        predicted = self.combined.predict(x=content, batch_size=self.settings["batch_size"])
        return predicted.tolist()

    def validate(self, counter, batches_dev):
        acc_mse = []
        for batch_dev in batches_dev:
            x_batch, y_batch = zip(*batch_dev)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            result = self.combined.predict(x_batch)
            mse = sklearn.metrics.mean_squared_error(result, y_batch)
            acc_mse.append(mse)
        print("%d Validation[Combined loss: %f]" % (counter, float(np.mean(np.array(acc_mse)))))
