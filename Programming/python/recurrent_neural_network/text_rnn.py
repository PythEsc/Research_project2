import tensorflow as tf


class TextRNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, lstm_size, lstm_layers, batch_size, learning_rate=0.01):
        # Place holders for input, output and dropout
        # self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Embedding layer
        self.embedding_vector = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
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
        W = tf.get_variable('W', [embedding_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.01, shape=[num_classes]), name='b')
        Wb = tf.matmul(self.outputs[:, -1], W) + b
        self.scores = tf.nn.softmax(Wb, name='scores')

        # Loss
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
        self.loss = tf.reduce_mean(losses)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Accuracy
        correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
