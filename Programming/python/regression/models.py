import os
import pickle
from collections import Iterable
from random import shuffle
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils import shuffle

from data_processing.emotion_mining.negation_handling import NegationHandler
from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage
from neural_networks.callbacks.nn_metrics import NNMetric
from neural_networks.nn_implementations import keras_TextNNBase
from neural_networks.nn_implementations.keras_TextCNN import TextCNN_Keras
from neural_networks.util import configuration_loader
from neural_networks.util.data_helpers import training_set_with_emotions_iter, validation_set_with_emotions_iter


class Regressor():
    def __init__(self, db: DataStorage, network: keras_TextNNBase, settings: dict, model='LinearRegression'):
        self.model = model
        self.db = db
        self.network = network
        self.settings = settings

        if self.model == 'LinearRegression':
            self.regressor = LinearRegression(fit_intercept=True, normalize=True)
        elif self.model == 'LogisticRegression':
            self.regressor = LogisticRegression(fit_intercept=True, solver='saga', multi_class='multinomial')
        else:
            raise ValueError('Unknown Model!')

        self.package_directory = os.path.dirname(os.path.abspath(__file__))

    def fit(self):
        # Generate batches
        precision = []
        recall = []
        f1 = []
        mse = []
        for train_tuple, valid_tuple in zip(self.training_batch_iterator(batch_size=10000),
                                            self.validation_batch_iterator(batch_size=1000)):
            x_train = train_tuple[0]
            y_train = train_tuple[1]
            emotes_train = train_tuple[2]

            del train_tuple

            x_dev = valid_tuple[0]
            y_dev = valid_tuple[1]
            emotes_dev = valid_tuple[2]

            del valid_tuple

            nn_out_train = self.network.predict(x=x_train)

            output = [np.hstack((nn_out_train[i], emotes_train[i])) for i in range(len(x_train))]

            self.regressor.fit(X=output, y=y_train)

            val_pred = self.predict(x_dev, emotions=emotes_dev)

            precision_value, recall_value, f1_value, mse_value = NNMetric.evaluate(val_predict=val_pred, val_targ=y_dev)
            precision.append(precision_value)
            recall.append(recall_value)
            f1.append(f1_value)
            mse.append(mse_value)
            print('Current precision: {}'.format(precision_value))
            print('Current recall: {}'.format(recall_value))
            print('Current f1-score: {}'.format(f1_value))
            print('Current MSE: {}'.format(mse_value))

        print("------------ Finished fit ------------")
        print('Avg. precision: {}'.format(np.mean(precision)))
        print('Avg. recall: {}'.format(np.mean(recall)))
        print('Avg. f1-score: {}'.format(np.mean(f1)))
        print('Avg. MSE: {}'.format(np.mean(mse)))
        pickle.dump(self.regressor, open(os.path.join(self.settings["checkpoint_path"], "regressor.sav"), 'wb'))

    def predict(self, content: np.array, emotions: Optional[np.array] = None) -> np.array:
        rnn_out = self.network.predict(content)

        if emotions is None:
            nh = NegationHandler(self.db)
            nh_out = []
            for x in content:
                nh_out.append(nh.get_emotion(x))
            emotions = np.array(nh_out, dtype=np.float32)

        # output = [np.hstack((out[i], nh_out[i])) for i in range(len(content))]
        output = [np.hstack((rnn_out[i], emotions[i])) for i in range(len(content))]

        y_pred_n = self.regressor.predict(output)
        y_pred = np.empty(y_pred_n.shape, dtype=np.float32)
        for i in range(len(y_pred_n)):
            y_pred[i] = np.array([(p - y_pred_n.min()) / (y_pred_n.max() - y_pred_n.min()) for p in y_pred_n[i]],
                                 dtype=np.float32)

        return y_pred

    def validate(self):
        precision = []
        recall = []
        f1 = []
        mse = []

        for batch_x, batch_y, batch_emotions in self.validation_batch_iterator(batch_size=10000):
            predicted = self.predict(batch_x, emotions=batch_emotions)

            precision_value, recall_value, f1_value, mse_value = NNMetric.evaluate(val_predict=predicted,
                                                                                   val_targ=batch_y)

            mse.append(mse_value)
            precision.append(precision_value)
            recall.append(recall_value)
            f1.append(f1_value)

        print("------------- Validation Finished -------------")
        print("Avg. MSE: %.4f (+/- %.4f)" % (float(np.mean(mse)), float(np.std(mse))))
        print("Avg. precision: %.4f (+/- %.4f)" % (float(np.mean(precision)), float(np.std(precision))))
        print("Avg. recall: %.4f (+/- %.4f)" % (float(np.mean(recall)), float(np.std(recall))))
        print("Avg. f1-score: %.4f (+/- %.4f)" % (float(np.mean(f1)), float(np.std(f1))))

    def load_checkpoint(self):
        self.regressor = pickle.load(open(os.path.join(self.settings["checkpoint_path"], "regressor.sav"), 'rb'))

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
    path = "../../results/CNN/augmented_1/"
    settings = configuration_loader.load_config(path + "learning_rate0.001_numlayer1_dropout0.3.json")
    settings["checkpoint_path"] = path + "checkpoints/"

    network = TextCNN_Keras(settings=settings)
    network.load_checkpoint()

    reg = Regressor(db=db, network=network, settings=settings)
    reg.fit()
    reg.validate()
