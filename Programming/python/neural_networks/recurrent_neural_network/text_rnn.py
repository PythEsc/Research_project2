import os
from time import time

import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.contrib import learn

from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage
from neural_networks.data_helpers import get_training_set
from neural_networks.neural_network import NeuralNetwork


class TextRNN(NeuralNetwork):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, lstm_size, lstm_layers, batch_size, learning_rate=0.01,
                 l2_reg_lambda=0.0):

        # Place holders for input, output and dropout
        # self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.embedding_vector = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name='embedding_vector')
        self.embedding_layer = tf.nn.embedding_lookup(self.embedding_vector, self.input_x)

        # Basic LSTM cell
        self.lstm = tf.contrib.rnn.BasicLSTMCell(embedding_size)
        # Dropout to cell
        self.drop = tf.contrib.rnn.DropoutWrapper(self.lstm, output_keep_prob=self.dropout_keep_prob)
        # Stack multiple layers of LSTM
        self.cell = tf.contrib.rnn.MultiRNNCell([self.drop] * lstm_layers)

        # Forward pass
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.embedding_layer,
                                                           initial_state=self.initial_state)

        # Output
        W = tf.get_variable('W',
                            shape=[embedding_size, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.01, shape=[num_classes]), name='b')
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        Wb = tf.matmul(self.outputs[:, -1], W) + b
        self.scores = tf.nn.softmax(Wb, name='scores')

        # Loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Accuracy
        # print(type(self.scores.eval()))
        # print(type(self.input_y.eval()))
        # correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
        # correct_predictions = metrics.mean_squared_error(self.input_y, self.scores)
        # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

        # self.accuracy = tf.metrics.mean_squared_error(self.input_y, self.scores)
        self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.input_y, self.scores))))

    @staticmethod
    def train(db: DataStorage, sample_percentage: float = 0.2, required_mse: float = 0.9):
        x_text, y = get_training_set(db)

        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = np.copy(y)
        for i in range(len(shuffle_indices)):
            value = y[shuffle_indices[i]]
            y_shuffled[i] = value

        # Split train/test set
        dev_sample_index = -1 * int(sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        # Training
        # ==================================================
        def batch_iter(data, batch_size, shuffle=True):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data) - 1) / batch_size)  # + 1
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                print("\rBatch iteration: %.2f%%" % (batch_num/num_batches_per_epoch*100), end='')
                yield shuffled_data[start_index:end_index]

        num_epochs = 20
        embedding_dim = 50
        lstm_size = 128
        lstm_layers = 1
        batch_size = 17
        learning_rate = 0.01

        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                # Create the TextRNN with the numbers of the created training set
                rnn = TextRNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=embedding_dim,
                    lstm_size=lstm_size,
                    lstm_layers=lstm_layers,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )

                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                iteration = 1
                last_e = 0
                t = time()
                error = required_mse + 1
                for e in range(num_epochs):

                    if time() - t > 1000:
                        break

                    state = sess.run(rnn.initial_state)

                    for ii, batch in enumerate(batch_iter(list(zip(x_train, y_train)), batch_size, e)):
                        x, y = zip(*batch)
                        feed = {rnn.input_x: x,
                                rnn.input_y: y,
                                rnn.dropout_keep_prob: 0.5,
                                rnn.initial_state: state}
                        loss, state, _, error = sess.run([rnn.loss, rnn.final_state, rnn.optimizer, rnn.accuracy],
                                                         feed_dict=feed)

                        # print("Step: {}".format(ii))

                        if last_e < e:
                            last_e = e
                            print("Epoch: {}/{}".format(e, num_epochs),
                                  "Iteration: {}".format(iteration),
                                  "Train loss: {:.3f}".format(loss),
                                  "Mean squared error: {:.3f}".format(error))

                        # if iteration % 25 == 0:
                        #     val_acc = []
                        #     val_state = sess.run(rnn.cell.zero_state(batch_size, tf.float32))
                        #     for batch in batch_iter(list(zip(x_dev, y_dev)), batch_size, num_epochs):
                        #         x, y = zip(*batch)
                        #         feed = {rnn.input_x: x,
                        #                 rnn.input_y: y,
                        #                 rnn.dropout_keep_prob: 1,
                        #                 rnn.initial_state: val_state}
                        #         batch_acc, val_state = sess.run([rnn.accuracy, rnn.final_state], feed_dict=feed)
                        #         val_acc.append(batch_acc)
                        #     print("Val acc: {:.3f}".format(np.mean(val_acc)))
                        iteration += 1

                    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                    checkpoint_dir = os.path.abspath("./checkpoints/")
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    saver.save(sess, "./checkpoints/rnn.ckpt")

                    if error < required_mse:
                        print("Reached the required mean squared error. Stop training")
                        break

                pred = []

                test_state = sess.run(rnn.cell.zero_state(batch_size, tf.float32))
                for ii, batch in enumerate(batch_iter(list(zip(x_dev, y_dev)), batch_size, 1)):
                    x, y = zip(*batch)
                    feed = {rnn.input_x: x,
                            rnn.input_y: y,
                            rnn.dropout_keep_prob: 1,
                            rnn.initial_state: test_state}

                    preds, test_state = sess.run([rnn.scores, rnn.final_state], feed_dict=feed)
                    pred.append(preds)

                pred = pred[0]
                y_dev = y_dev[:len(pred)]

                mae = metrics.mean_absolute_error(y_dev, pred)
                mse = metrics.mean_squared_error(y_dev, pred)

                print('\nMean absolute error: {}'.format(mae))
                print('Mean squared error: {}'.format(mse))

    @staticmethod
    def predict(content: str) -> list:
        pass


if __name__ == '__main__':
    db = MongodbStorage()

    TextRNN.train(db)
