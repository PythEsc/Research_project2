from __future__ import print_function, division

import numpy as np
import os
import sklearn.metrics
from keras.layers import Activation
from keras.layers import Input, Dense, Reshape, Flatten, MaxPooling2D, Embedding, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.contrib import learn

from importer.database.mongodb import MongodbStorage
from neural_networks.util import data_helpers
from neural_networks.util.data_helpers import get_training_set, clean_text
from pre_trained_embeddings.word_embeddings import WordEmbeddings


class TextCNN_Keras():
    def __init__(self, settings: dict):
        self.settings = settings
        self.vocab_processor = None

        # Read in the data
        self.x_train, self.x_dev, self.y_train, self.y_dev = self.read_data()
        # Get the dimensions for the network by reading the data
        self.sequence_length = self.x_train.shape[1]
        self.num_classes = self.y_train.shape[1]
        self.vocab_size = len(self.vocab_processor.vocabulary_)

        # Initialize the network
        # Start with creating placeholder variables
        self.input_x = Input((self.sequence_length,), name="input_x")
        self.input_y = Input((self.num_classes,), name="input_y")

        optimizer = Adam(lr=0.0005, beta_1=0.5)

        # Build and compile the generator
        self.cnn = self.build_cnn()
        self.cnn.compile(loss="mse", metrics=['accuracy'], optimizer=optimizer)

    def load_checkpoint(self):
        from keras.models import load_model
        self.cnn = load_model("../../checkpoints/checkpoint_gp_wgan/model/gp_wgan/latest_model.h5")

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
        # model.add(Dropout(self.dropout_keep_prob))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.summary()

        result = model(self.input_x)

        return Model(self.input_x, result)

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

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = np.array(y)
        for i in range(len(shuffle_indices)):
            value = y[shuffle_indices[i]]
            y_shuffled[i] = value

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(self.settings["dev_sample_percentage"] * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        return x_train, x_dev, y_train, y_dev

    def train(self):

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(self.x_train, self.y_train)), self.settings["batch_size"], self.settings["num_epochs"])
        batches_dev = data_helpers.batch_iter(
            list(zip(self.x_dev, self.y_dev)), self.settings["batch_size"], 1)
        counter = 1
        for batch in batches:

            # ---------------------
            #  Train Generator
            # ---------------------
            x_batch, y_batch = zip(*batch)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            # Train the generator
            cnn_loss = self.cnn.train_on_batch(x_batch, y_batch)

            # Plot the progress
            if counter % 20 == 0:
                print("%d Training[CNN loss: %f, CNN accuracy: %f]" % (counter, cnn_loss[0], cnn_loss[1]))

            # If at save interval => save generated image samples
            if counter % 500 == 0:
                path = "../results/keras_model/"
                os.makedirs(path, exist_ok=True)
                self.cnn.save(os.path.join(path, "three_layer_nomaxpooljustattheend_latest_model.h5"), include_optimizer=False)
                self.cnn.save_weights(os.path.join(path, "three_layer_nomaxpooljustattheend_latest_model_weights.h5"), True)
                self.validate(counter, batches_dev)
            counter += 1

    def predict(self, content: list) -> list:
        predicted = self.cnn.predict(x=content, batch_size=self.settings["batch_size"])
        return predicted.tolist()

    def validate(self, counter, batches_dev):
        acc_mse = []
        for batch_dev in batches_dev:
            x_batch, y_batch = zip(*batch_dev)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            result = self.cnn.predict(x_batch)
            mse = sklearn.metrics.mean_squared_error(result, y_batch)
            acc_mse.append(mse)
        print("%d Validation[CNN loss: %f]" % (counter, float(np.mean(np.array(acc_mse)))))
