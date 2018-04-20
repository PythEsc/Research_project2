import os
from abc import ABC, abstractmethod
from collections import Iterable

import numpy as np
import sklearn
from keras import Input, losses
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.contrib import learn

from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage
from neural_networks.callbacks.nn_metrics import NNMetric
from neural_networks.util.data_helpers import training_set_iter
from pre_trained_embeddings.word_embeddings import WordEmbeddings


class TextNN_Keras(ABC):

    def __init__(self, settings: dict):
        self.settings = settings
        self.vocab_processor = None
        self.checkpoint_path = settings["checkpoint_path"]

        # Initialize vocabulary processor
        max_document_length, num_classes = self.initialize_vocab()

        # Get the dimensions for the network by reading the data
        self.sequence_length = max_document_length
        self.num_classes = num_classes
        self.vocab_size = len(self.vocab_processor.vocabulary_)

        # Initialize the network
        # Start with creating placeholder variables
        self.input_x = Input((self.sequence_length,), name="input_x")
        self.input_y = Input((self.num_classes,), name="input_y")

        optimizer = Adam(lr=self.settings["learning_rate"], beta_1=0.5)

        # Build and compile the generator
        self.nn = self.build()
        self.nn.compile(loss=losses.mean_squared_error,
                        optimizer=optimizer)
        self.callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.005, patience=8),
                          ModelCheckpoint(filepath=self.settings["checkpoint_path"] + "/nn_best.model",
                                          save_best_only=True),
                          ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=0.0005), NNMetric()]

    @abstractmethod
    def build(self):
        pass

    def initialize_vocab(self):
        # Data Preparation
        # ==================================================

        db = MongodbStorage()

        # Create feature matrix using vocab from trained WordEmbeddings
        print('Creating feature matrix...')
        max_document_length = None
        min_document_length = None
        num_classes = None
        total_document_length = 0
        counter = 0

        for counter, set_entry in enumerate(training_set_iter(db=db)):
            text = set_entry[0]
            reactions = set_entry[1]

            num_classes = len(reactions)

            length = len(text.split(" "))
            if max_document_length is None or length > max_document_length:
                max_document_length = length
            if min_document_length is None or length < min_document_length:
                min_document_length = length
            total_document_length += length

        mean_document_length = total_document_length / counter
        used_document_length = min(int(mean_document_length * 1.5), max_document_length)

        print('Max document length is {}.'.format(max_document_length))
        print('Mean document length is {}.'.format(mean_document_length))
        print('Min document length is {}.'.format(min_document_length))
        print('Used document length is {}'.format(used_document_length))
        print('Total number of documents is {}'.format(counter))

        embedding_dim = 50
        we = WordEmbeddings(embedding_dim)

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(
            max_document_length=used_document_length,
            vocabulary=we.categorical_vocab)

        return used_document_length, num_classes

    def load_checkpoint(self):
        from keras.models import load_model
        self.nn = load_model(self.checkpoint_path + "/nn_best.model")

    def train(self):
        # Generate batches
        db = MongodbStorage()
        for batch_x, batch_y in self.batch_iterator(db=db, batch_size=5000):
            os.makedirs(self.settings["checkpoint_path"], exist_ok=True)
            self.nn.fit(batch_x, batch_y, batch_size=self.settings["batch_size"], epochs=self.settings["num_epochs"],
                        validation_split=self.settings["dev_sample_percentage"], shuffle=True, verbose=2,
                        callbacks=self.callbacks)

    def predict(self, content: list) -> list:
        predicted = self.nn.predict(x=content, batch_size=self.settings["batch_size"])
        return predicted.tolist()

    def validate(self, counter, batches_dev):
        acc_mse = []
        for batch_dev in batches_dev:
            x_batch, y_batch = zip(*batch_dev)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            result = self.nn.predict(x_batch)
            mse = sklearn.metrics.mean_squared_error(result, y_batch)
            acc_mse.append(mse)
        print("%d Validation[CNN loss: %f]" % (counter, float(np.mean(np.array(acc_mse)))))

    def batch_iterator(self, db: DataStorage, batch_size: int, threshold: int = 1, use_likes: bool = False) -> Iterable:
        x_batch = None
        y_batch = None

        for index, set_entry in enumerate(training_set_iter(db=db, threshold=threshold, use_likes=use_likes,
                                                            max_post_length=self.sequence_length)):

            x = set_entry[0]
            y = np.array(set_entry[1], dtype=np.float32)

            x = np.array(list(self.vocab_processor.transform([x]))[0], dtype=np.int32)

            if x_batch is None:
                x_batch = np.empty((batch_size, len(x)), dtype=np.int32)
            if y_batch is None:
                y_batch = np.empty((batch_size, len(y)), dtype=np.float32)

            x_batch[index % batch_size] = x
            y_batch[index % batch_size] = y

            if index % batch_size == batch_size - 1:
                x_batch, y_batch = shuffle(x_batch, y_batch)
                yield x_batch, y_batch
                x_batch = None
                y_batch = None

        if x_batch is not None and y_batch is not None:
            x_batch, y_batch = shuffle(x_batch, y_batch)
            yield x_batch, y_batch
