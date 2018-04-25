import os
import pickle
from collections import Iterable
from random import shuffle

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from data_processing.emotion_mining.negation_handling import NegationHandler
from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage
from neural_networks.nn_implementations import keras_TextNNBase
from neural_networks.nn_implementations.keras_TextCNN import TextCNN_Keras
from neural_networks.util.configuration_loader import DEFAULT_DICT
from neural_networks.util.data_helpers import training_set_with_emotions_iter, validation_set_with_emotions_iter


class Regressor():
    def __init__(self, db: DataStorage, network: keras_TextNNBase, settings: dict, model='LinearRegression'):
        self.model = model
        self.db = db
        self.network = network
        self.settings = settings

        if self.model == 'LinearRegression':
            self.regressor = LinearRegression(fit_intercept=True, normalize=True)
        else:
            raise ValueError('Unknown Model!')

        self.package_directory = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.package_directory, 'model.sav')

    def fit(self):
        # Generate batches
        y_pred = []
        for train_tuple, valid_tuple in zip(self.training_batch_iterator(batch_size=5000),
                                            self.validation_batch_iterator(batch_size=500)):
            x_train = train_tuple[0]
            y_train = train_tuple[1]
            emotes_train = train_tuple[2]

            x_dev = valid_tuple[0]
            y_dev = valid_tuple[1]
            emotes_dev = valid_tuple[2]

            nn_out_train = self.network.predict(x=x_train)
            nn_out_dev = self.network.predict(x=x_dev)

            output = [np.hstack((nn_out_train[i], emotes_train[i])) for i in range(len(x_train))]
            output_dev = [np.hstack((nn_out_dev[i], emotes_dev[i])) for i in range(len(x_dev))]

            print('regression')
            self.regressor.fit(X=output, y=y_train)

            y_pred_n = self.regressor.predict(output_dev)

            for i in range(len(y_pred_n)):
                y_pred.append([(p + 1) / 2 for p in y_pred_n[i]])

            mse = metrics.mean_squared_error(y_dev, y_pred)
            print('MSE: {}'.format(mse))

        pickle.dump(self.regressor, open(self.model_path, 'wb'))

    def predict(self, content: list) -> list:
        self.regressor = pickle.load(open(self.model_path, 'rb'))

        rnn_out = self.network.predict(content)

        nh = NegationHandler(self.db)
        nh_out = []
        for x in content:
            nh_out.append(nh.get_emotion(x))

        # output = [np.hstack((out[i], nh_out[i])) for i in range(len(content))]
        output = [np.hstack((rnn_out[i], nh_out[i])) for i in range(len(content))]

        y_pred_n = self.regressor.predict(output)
        y_pred = []
        for i in range(len(y_pred_n)):
            y_pred.append([(p + 1) / 2 for p in y_pred_n[i]])
        return y_pred

    def training_batch_iterator(self, batch_size: int, threshold: int = 1, use_likes: bool = False) -> Iterable:
        x_batch = None
        y_batch = None
        emotions_batch = None

        index = 0
        for index, set_entry in enumerate(
                training_set_with_emotions_iter(db=self.db, threshold=threshold, use_likes=use_likes,
                                                max_post_length=self.network.sequence_length,
                                                allow_augmented_data=self.settings["allow_augmented_data"])):

            x = set_entry[0]
            y = np.array(set_entry[1], dtype=np.float32)
            emotions = np.array(set_entry[2], dtype=np.float32)

            x = np.array(list(self.network.vocab_processor.transform([x]))[0], dtype=np.int32)

            if x_batch is None:
                x_batch = np.empty((batch_size, len(x)), dtype=np.int32)
            if y_batch is None:
                y_batch = np.empty((batch_size, len(y)), dtype=np.float32)
            if emotions_batch is None:
                emotions_batch = np.empty((batch_size, len(emotions)), dtype=np.float32)

            x_batch[index % batch_size] = x
            y_batch[index % batch_size] = y
            emotions_batch[index % batch_size] = emotions

            if index % batch_size == batch_size - 1:
                x_batch, y_batch, emotions_batch = shuffle(x_batch, y_batch, emotions_batch)
                yield x_batch, y_batch, emotions_batch
                x_batch = None
                y_batch = None
                emotions_batch = None

        if x_batch is not None and y_batch is not None and emotions_batch is not None:
            x_batch = x_batch[:(index % batch_size) + 1]
            y_batch = y_batch[:(index % batch_size) + 1]
            emotions_batch = emotions_batch[:(index % batch_size) + 1]

            x_batch, y_batch, emotions_batch = shuffle(x_batch, y_batch, emotions_batch)
            yield x_batch, y_batch, emotions_batch

    def validation_batch_iterator(self, batch_size: int, threshold: int = 1,
                                  use_likes: bool = False) -> Iterable:
        x_batch = None
        y_batch = None
        emotions_batch = None

        index = 0
        for index, set_entry in enumerate(
                validation_set_with_emotions_iter(db=self.db, threshold=threshold, use_likes=use_likes,
                                                  max_post_length=self.network.sequence_length,
                                                  allow_augmented_data=self.settings["allow_augmented_data"])):

            x = set_entry[0]
            y = np.array(set_entry[1], dtype=np.float32)
            emotions = np.array(set_entry[2], dtype=np.float32)

            x = np.array(list(self.network.vocab_processor.transform([x]))[0], dtype=np.int32)

            if x_batch is None:
                x_batch = np.empty((batch_size, len(x)), dtype=np.int32)
            if y_batch is None:
                y_batch = np.empty((batch_size, len(y)), dtype=np.float32)
            if emotions_batch is None:
                emotions_batch = np.empty((batch_size, len(emotions)), dtype=np.float32)

            x_batch[index % batch_size] = x
            y_batch[index % batch_size] = y
            emotions_batch[index % batch_size] = emotions

            if index % batch_size == batch_size - 1:
                x_batch, y_batch, emotions_batch = shuffle(x_batch, y_batch, emotions_batch)
                yield x_batch, y_batch, emotions_batch
                x_batch = None
                y_batch = None
                emotions_batch = None

        if x_batch is not None and y_batch is not None and emotions_batch is not None:
            x_batch = x_batch[:(index % batch_size) + 1]
            y_batch = y_batch[:(index % batch_size) + 1]
            emotions_batch = emotions_batch[:(index % batch_size) + 1]

            x_batch, y_batch, emotions_batch = shuffle(x_batch, y_batch, emotions_batch)
            yield x_batch, y_batch, emotions_batch


if __name__ == '__main__':
    db = MongodbStorage()
    settings = DEFAULT_DICT

    network = TextCNN_Keras(settings=settings)
    network.load_checkpoint()

    reg = Regressor(db=db, network=network, settings=DEFAULT_DICT)
    reg.fit()

    content = [
        'This is me.'
    ]

    print(reg.predict(content))
