import tensorflow as tf
from importer.data_importer import DataImporter
import numpy as np
from tensorflow.contrib import learn
from sklearn import metrics
import itertools
from time import time

import os
import datetime

from recurrent_neural_network.text_rnn import TextRNN
from python.convolutional_neural_network.data_helpers import get_training_set
from python.importer.database.mongodb import MongodbStorage

# Load the data
print("Loading data...")

importer_sainsbury = DataImporter("../../../data/Filtered/Sainsbury.zip", "../../../data/Unzipped/Sainsbury")
importer_sainsbury.load()
x_text, y = importer_sainsbury.prepare_data_for_CNN()

importer_tesco = DataImporter("../../../data/Filtered/Tesco.zip", "../../../data/Unzipped/Tesco")
importer_tesco.load()
x_text2, y2 = importer_tesco.prepare_data_for_CNN()

for i, x in enumerate(x_text2):
    x_text.append(x)
    y.append(y2[i])

# db = MongodbStorage()
# x_text, y = get_training_set(db)

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
dev_sample_percentage = 0.2
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size)# + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

num_epochs = 20
embedding_dim = 50
lstm_size=128
lstm_layers=1
batch_size = 17
learning_rate=0.01

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
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
        for e in range(num_epochs):

            if time() - t > 1000:
                continue

            state = sess.run(rnn.initial_state)

            for ii, batch in enumerate(batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)):
                x, y = zip(*batch)
                feed = {rnn.input_x: x,
                        rnn.input_y: y,
                        rnn.dropout_keep_prob: 0.5,
                        rnn.initial_state: state}
                loss, state, _, error = sess.run([rnn.loss, rnn.final_state, rnn.optimizer, rnn.accuracy], feed_dict=feed)


                # print("Step: {}".format(ii))

                if last_e < e:
                    last_e = e
                    print("Epoch: {}/{}".format(e, num_epochs),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.3f}".format(loss),
                          "Mean squared error: {:.3f}".format(error) )

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
        saver.save(sess, "./checkpoints/rnn.ckpt")

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
