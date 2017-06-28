import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from importer.database.database_access import DataStorage
from neural_networks.neural_network import NeuralNetwork


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
            self.accuracy = tf.losses.mean_squared_error(self.input_y, self.scores)

    @staticmethod
    def train(db: DataStorage, sample_percentage: float = 0.2, required_mse: float = 0.3, restore=False):
        # TODO: Add some code to save/restore the model. At the moment we always have to start from the beginning when
        # training is stopped
        pass

    @staticmethod
    def predict(content: list) -> list:
        """
        This method predicts the Facebook reactions for a single post

        :param content: The content of a single Facebook post but as a list ["..text..."]. This can also be used for the
                        batch prediction later on. 
        :return: A list of lists containing the ratio of reactions ['LIKE', 'LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY', 'THANKFUL']
        """
        # CHECKPOINT NEEDS TO BE LATEST CHECKPOINT YOU TRAINED
        package_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint_for_evaluation = os.path.join(package_directory, "runs/1498554852/checkpoints/")

        checkpoint_file = tf.train.latest_checkpoint(checkpoint_for_evaluation)
        vocab_path = os.path.join(checkpoint_for_evaluation, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        content = np.array(list(vocab_processor.transform(content)))
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/scores").outputs[0]

                # Collect the predictions here
                result = sess.run(predictions, {input_x: content, dropout_keep_prob: 1})
                return result

if __name__ == '__main__':

    from importer.database.mongodb import MongodbStorage
    db = MongodbStorage()

    # TextRNN.train(db, required_mse=0.1, restore=False)
    # TextRNN.train(db, required_mse=0.1, restore=True)


    content = [
        'This is me.',
        'I hate your supermarket!',
        'Your employees are so rude!',
    ]

    predicted_reactions = TextCNN.predict(content)

    for i, r in enumerate(content):
        print('{}:\n{}'.format(r, predicted_reactions[i]))
