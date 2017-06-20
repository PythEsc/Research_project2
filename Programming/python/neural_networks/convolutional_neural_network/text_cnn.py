import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage
from neural_networks.data_helpers import get_training_set, batch_iter
from neural_networks.neural_network import NeuralNetwork
from pre_trained_embeddings.word_embeddings import WordEmbeddings


class TextCNN(NeuralNetwork):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, embedding=None):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if embedding is None:
                self.embedding_vector = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name='embedding_vector')
                self.embedding_layer = tf.nn.embedding_lookup(self.embedding_vector, self.input_x)
            else:
                self.embedding_vector = tf.Variable(embedding, name='embedding_vector')
                self.embedding_layer = tf.nn.embedding_lookup(self.embedding_vector, self.input_x)

            self.embedded_chars_expanded = tf.expand_dims(self.embedding_layer, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
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
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            Wb = tf.matmul(self.h_drop, W) + b
            self.scores = tf.nn.softmax(Wb, name="scores")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.input_y, self.scores))), name="accuracy")

    @staticmethod
    def train(db: DataStorage, sample_percentage: float = 0.2, required_mse: float = 0.3):
        # TODO: Add some code to save/restore the model. At the moment we always have to start from the beginning when
        # training is stopped
        pass

    @staticmethod
    def predict(content: str) -> list:
        """
        This method predicts the Facebook reactions for a single post

        :param content: The content of a single Facebook post
        :return: A list containing the ratio of reactions ['LIKE', 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY', 'THANKFUL']
        """
        pass
