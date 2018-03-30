import os

import datetime
import time

import numpy as np
import tensorflow as tf

from importer.database.database_access import DataStorage
from neural_networks.neural_network import NeuralNetwork
from tensorflow.contrib import learn

from importer.database.mongodb import MongodbStorage
from neural_networks import data_helpers
from neural_networks.data_helpers import get_training_set, clean_text
from pre_trained_embeddings.word_embeddings import WordEmbeddings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TextCNN(NeuralNetwork):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, embedding=None, flags=None):
        # Placeholders for input, output and dropout
        self.FLAGS = flags
        self.vocab_processor = None

        self.filter_sizes = list(map(int, self.FLAGS.filter_sizes.split(",")))
        self.embedding_size = self.FLAGS.embedding_dim
        self.num_filters = self.FLAGS.num_filters
        self.l2_reg_lambda = self.FLAGS.l2_reg_lambda
        # Read in the data
        self.x_train, self.x_dev, self.y_train, self.y_dev = self.read_data()
        # Get the dimensions for the network by reading the data
        self.sequence_length = self.x_train.shape[1]
        self.num_classes = self.y_train.shape[1]
        self.vocab_size = len(self.vocab_processor.vocabulary_)

        # Initialize the network
        # Start with creating placeholder variables
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if embedding is None:
                self.embedding_vector = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name='embedding_vector')
                self.embedding_layer = tf.nn.embedding_lookup(self.embedding_vector, self.input_x)
            else:
                self.embedding_vector = tf.Variable(embedding, name='embedding_vector')
                self.embedding_layer = tf.nn.embedding_lookup(self.embedding_vector, self.input_x)

            self.embedded_chars_expanded = tf.expand_dims(self.embedding_layer, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            Wb = tf.matmul(self.h_drop, W) + b
            self.scores = tf.nn.softmax(Wb, name="scores")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.accuracy = tf.losses.mean_squared_error(self.input_y, self.scores)

        # Optimizer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.0005)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS.num_checkpoints)


    def train(self, sample_percentage: float = 0.2, required_mse: float = 0.3, restore=False):
        # Training
        # ==================================================
        experiment_file = './experiments/pre_trained.csv'
        with open(experiment_file, 'w+', encoding='utf-8') as file:
            file.write('val_mse,train_mse')

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.FLAGS.allow_soft_placement,
                log_device_placement=self.FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Define Training procedure
                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in self.grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", self.loss)
                acc_summary = tf.summary.scalar("accuracy", self.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)


                # Initialize all variables
                self.sess.run(tf.global_variables_initializer())

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                # Write vocabulary
                self.vocab_processor.save(os.path.join(out_dir, "vocab"))

                # Training loop. For each batch...
                errors = []
                for batch in batches:
                    # ---------------------------------------
                    # -------------TRAIN STEP ---------------
                    # ---------------------------------------
                    # Get Batch
                    x_batch, y_batch = zip(*batch)
                    # Create feed dict for training step
                    feed_dict = self.get_feed_dict_train(x_batch, y_batch)
                    # Do one forward pass and optimize in one session
                    _, step, summaries, loss, accuracy = self.sess.run(
                        [self.train_op, self.global_step, train_summary_op, self.loss, self.accuracy],
                        feed_dict)
                    # Log the accuracy
                    errors.append(accuracy)
                    # Write Tensorboard summary
                    train_summary_writer.add_summary(summaries, step)
                    # Update the global step
                    current_step = tf.train.global_step(self.sess, self.global_step)

                    # --------------------------------------------
                    # -------------EVALUATION STEP ---------------
                    # --------------------------------------------
                    if current_step % self.FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        # Create feed dict for evaluation step
                        dev_feed_dict = self.get_feed_dict_dev(self.x_dev, self.y_dev)
                        # Do one forward pass and optimize in one session
                        step, summaries, loss, accuracy = self.sess.run(
                            [self.global_step, dev_summary_op, self.loss, self.accuracy],
                            dev_feed_dict)
                        val_error = accuracy
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                        if dev_summary_writer:
                            dev_summary_writer.add_summary(summaries, step)
                        print("")

                        with open(experiment_file, 'a', encoding='utf-8') as file:
                            file.write('\n{},{}'.format(val_error, np.mean(errors)))

                        errors = []

                    # Save network
                    if current_step % self.FLAGS.checkpoint_every == 0:
                        path = self.saver.save(self.sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

    def get_feed_dict_train(self, x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: self.FLAGS.dropout_keep_prob
        }

        return feed_dict

    def get_feed_dict_dev(self, x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: 1.0
        }

        return feed_dict

    def predict_emotions(self, samples: list):
        return self.sess.run([self.scores], {self.input_x: samples,
                                             self.dropout_keep_prob: self.FLAGS.dropout_keep_prob})

    @staticmethod
    def predict(samples: list, checkpoint_path="runs/1497955024/checkpoints/") -> list:
        """
        This method predicts the Facebook reactions for a single post

        :param samples: The content of a single Facebook post but as a list ["..text..."]. This can also be used for the
                        batch prediction later on.
        :param checkpoint_path: Checkpoint path
        :return: A list of lists containing the ratio of reactions ['LIKE', 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY', 'THANKFUL']
        """
        package_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint_for_evaluation = os.path.join(package_directory, checkpoint_path)

        checkpoint_path = tf.train.latest_checkpoint(checkpoint_for_evaluation)
        vocab_path = os.path.join(checkpoint_for_evaluation, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        samples = np.array(list(vocab_processor.transform(samples)))
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))
                saver.restore(sess, checkpoint_path)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/scores").outputs[0]

                # Collect the predictions here
                result = sess.run(predictions, {input_x: samples, dropout_keep_prob: 1})
                return result
