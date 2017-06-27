import os
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.contrib import learn

from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage
from neural_networks.data_helpers import get_training_set, batch_iter, clean_text
from neural_networks.neural_network import NeuralNetwork
from pre_trained_embeddings.word_embeddings import WordEmbeddings
from neural_networks.model_save_functions import save_model, restore_model


class TextRNN(NeuralNetwork):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, lstm_layers, batch_size, learning_rate=0.001,
                 l2_reg_lambda=0.0, embedding=None):

        # Place holders for input, output and dropout
        # self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        if embedding is None:
            self.embedding_vector = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name='embedding_vector')
            self.embedding_layer = tf.nn.embedding_lookup(self.embedding_vector, self.input_x)
        else:
            self.embedding_vector = tf.Variable(embedding, name='embedding_vector')
            self.embedding_layer = tf.nn.embedding_lookup(self.embedding_vector, self.input_x)

        # Basic LSTM cell
        self.lstm = tf.contrib.rnn.BasicLSTMCell(embedding_size)
        # Dropout to cell
        self.drop = tf.contrib.rnn.DropoutWrapper(self.lstm, output_keep_prob=self.dropout_keep_prob)
        # Stack multiple layers of LSTM
        self.cell = tf.contrib.rnn.MultiRNNCell([self.drop] * lstm_layers)

        # Input length
        self.l = tf.cast(tf.reduce_sum(tf.sign(tf.abs(self.input_x)), reduction_indices=1), tf.int32)

        # Forward pass
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        # self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.embedding_layer,
        #                                                    initial_state=self.initial_state)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, self.embedding_layer, dtype=tf.float32
                                                           , sequence_length=self.l)

        # Output
        W = tf.get_variable('W',
                            shape=[embedding_size, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.01, shape=[num_classes]), name='b')
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        # Wb = tf.matmul(self.outputs[:, -1], W) + b
        Wb = tf.matmul(self.outputs[:, -1], W) + b
        self.scores = tf.nn.softmax(Wb, name='scores')

        # Loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Accuracy
        self.accuracy = tf.losses.mean_squared_error(self.input_y, self.scores)

    @staticmethod
    def train(db: DataStorage, sample_percentage: float = 0.1, required_mse: float = 0.3, restore=False):
        print('Started training...')

        # Load data
        print('\nLoading training data...')
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
        vocab_processor = learn.preprocessing.VocabularyProcessor(mean_document_length, vocabulary=we.categorical_vocab)
        x = np.array(list(vocab_processor.transform(x_text)))
        y = np.copy(y)
        words_not_in_we = len(vocab_processor.vocabulary_) - len(we.vocab)
        if words_not_in_we > 0:
            print("Words not in pre-trained vocab: {:d}".format(words_not_in_we))

        # Randomly shuffle data
        np.random.seed(30)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = np.copy(y)
        for i in range(len(shuffle_indices)):
            value = y[shuffle_indices[i]]
            y_shuffled[i] = value

        # Split train/dev set
        dev_sample_index = -1 * int(sample_percentage * float(len(y)))
        x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
        y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
        print("\nTrain/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        # Training
        # ==================================================
        experiment_file = './experiments/pre_trained.csv'
        with open(experiment_file, 'w+', encoding='utf-8') as file:
            file.write('epoch,val_mse,train_mse')
        num_epochs = 30
        lstm_layers = 1
        batch_size = 200
        learning_rate = 0.0001

        print('\nStarting training...\n')
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                # Create the TextRNN with the numbers of the created training set
                rnn = TextRNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=embedding_dim,
                    lstm_layers=lstm_layers,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    embedding=we.embd
                )

                saver = tf.train.Saver()
                if restore:
                    print('restore')
                    restore_model(sess, './checkpoints/')
                else:
                    print('no restore')
                    sess.run(tf.global_variables_initializer())

                epoch = 1
                num_batches_per_epoch = int((len(x_train) - 1) / batch_size)

                state = sess.run(rnn.initial_state)

                errors = []
                for i, batch in enumerate(
                        batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs, shuffle=True)):
                    x, y = zip(*batch)
                    feed = {rnn.input_x: x,
                            rnn.input_y: y,
                            rnn.dropout_keep_prob: 0.5,
                            # rnn.initial_state: state
                            }
                    # loss, state, _, error = sess.run([rnn.loss, rnn.final_state, rnn.optimizer, rnn.accuracy],
                    #                                  feed_dict=feed)
                    loss, _, error = sess.run([rnn.loss, rnn.optimizer, rnn.accuracy], feed_dict=feed)

                    errors.append(error)

                    progress = i / (num_epochs * num_batches_per_epoch) * 100
                    print("\rBatch progress: %.2f%%" % progress, end='')

                    if (i + 1) % num_batches_per_epoch == 0:
                        train_mserror = np.mean(errors)
                        train_std = np.std(errors)
                        print("\nEpoch: {}/{}".format(epoch, num_epochs))
                        print("Train loss: {:.3f}".format(loss))
                        print("Training set - average Mean squared error: {:.4f}".format(train_mserror))
                        print("Training set - Mean squared error std: {:.4f}".format(train_std))

                        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                        checkpoint_dir = os.path.abspath("./checkpoints/")
                        if not os.path.exists(checkpoint_dir):
                            os.makedirs(checkpoint_dir)
                        save_model(saver, sess, "./checkpoints/rnn.ckpt")
                        # saver.save(sess, "./checkpoints/rnn.ckpt")
                        vocab_processor.save(os.path.join("./checkpoints", "vocab"))

                        # Validation accuracy
                        errors = []
                        feed = {rnn.input_x: x_dev,
                                rnn.input_y: y_dev,
                                rnn.dropout_keep_prob: 1}
                        error = sess.run([rnn.accuracy], feed_dict=feed)
                        errors.append(error)
                        mse = np.mean(errors)
                        with open(experiment_file, 'a', encoding='utf-8') as file:
                            file.write('\n{},{},{}'.format(epoch, np.mean(errors), train_mserror))

                        if train_mserror < required_mse:
                            print("Reached the required mean squared error. Stop training")
                            break

                        epoch += 1
                        errors = []

                pred = []

                # test_state = sess.run(rnn.cell.zero_state(batch_size, tf.float32))

                errors = []
                for batch in batch_iter(list(zip(x_dev, y_dev)), batch_size, 1):
                    x, y = zip(*batch)
                    feed = {rnn.input_x: x,
                            rnn.input_y: y,
                            rnn.dropout_keep_prob: 1,
                            # rnn.initial_state: test_state
                            }
                    error = sess.run([rnn.accuracy], feed_dict=feed)
                    errors.append(error)
                    # preds, test_state = sess.run([rnn.scores, rnn.final_state], feed_dict=feed)
                    # preds = sess.run([rnn.scores], feed_dict=feed)
                    # for p in preds[0]:
                    #     pred.append(p)

                # y_dev = y_dev[:len(pred)]

                # mae = metrics.mean_absolute_error(y_dev, pred)
                # mse = metrics.mean_squared_error(y_dev, pred)
                mse = np.mean(errors)
                # print('\nMean absolute error: {:.4f}'.format(mae))
                print('Testing set - average Mean squared error: {:.4f}'.format(mse))
                print("Testing set - Mean squared error std: {:.4f}".format(np.std(errors)))

    @staticmethod
    def predict(content: list) -> list:
        verbose = True
        if verbose:
            print('\nPredicting reactions for \"{}\" ...'.format(content))
            T = time()

        # Map data into vocabulary
        if verbose:
            print('\nLoading saved vocabulary...')
            t = time()
        vocab_path = os.path.join("./checkpoints", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x = np.array(list(vocab_processor.transform(clean_text(content))))
        if verbose: print('Loaded in {:.3f}s.'.format(time() - t))

        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            with sess.as_default():
                # Load the saved meta graph and restore variables
                if verbose:
                    print('\nLoading saved model...')
                    t = time()
                saver = tf.train.import_meta_graph("{}.meta".format("./checkpoints/rnn.ckpt"))
                saver.restore(sess, "./checkpoints/rnn.ckpt")
                if verbose: print('Model loaded in {:.3f}s.'.format(time() - t))

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("scores").outputs[0]

                # Collect the predictions here
                if verbose:
                    print('\nFeedforward through the network...')
                    t = time()
                predict = sess.run(predictions, {input_x: x, dropout_keep_prob: 1})
                if verbose: print('Pass through the network in {:.3f}s.'.format(time() - t))

                if verbose: print('\nTotal prediction in {:.3f}s.\n'.format(time() - T))

                # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                # for v in variables:
                #     print('\n{}'.format(v))
                #     print(v.eval())


                return predict[:len(content)]


if __name__ == '__main__':
    db = MongodbStorage()

    TextRNN.train(db, required_mse=0.1, restore=False)
    # TextRNN.train(db, required_mse=0.1, restore=True)


    content = ["This is just some sample post. I will try to use real posts later. Your salad is really disgusting!!!",
               "I really love to shop at Tesco!",
               "Your employees are so rude!",
               "Brought two packets of 'Snackers' chicken bites 60g from Ely store. "
               "The two packets felt very different in weight so I weighed them (unopened). "
               "One was 62g however the other was only 40g. Is this how Aldi keeps costs low, "
               "by only filling packets up by two-thirds?"]

    predicted_reactions = TextRNN.predict(content)

    for i, r in enumerate(content):
        print('{}:\n{}'.format(r, predicted_reactions[i]))

    # Sound when finished
    import platform

    if platform.system() == 'Windows':
        import winsound

        for _ in range(5):
            winsound.Beep(500, 200)
