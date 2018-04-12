import tensorflow as tf

from neural_networks.convolutional_neural_network.keras_TextCNN import TextCNN_Keras
from neural_networks.convolutional_neural_network.keras_TextRNN import TextRNN_Keras
from neural_networks.convolutional_neural_network.keras_TextRNNCNN import TextRNNCNN_Keras
from neural_networks.convolutional_neural_network.text_cnn import TextCNN

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 40, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("lstm_layers", 2, "LSTM Layer (default: 2)")
tf.flags.DEFINE_float("convolution_layers", 1, "LSTM Layer (default: 2)")
tf.flags.DEFINE_float("use_bn", False, "Add BN Layer (default: False)")
tf.flags.DEFINE_float("use_max_pooling", True, "Add MaxPool Layer (default: True)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 455, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def train_and_evaluate_cnn():
    # Start mongo with "sudo service mongod start"
    cnn = TextRNNCNN_Keras(flags=FLAGS)
    cnn.train()
    # Predict a little sample
    content = [
        'This is me.',
        'I hate your supermarket!',
        'Your employees are so rude!',
    ]
    predicted_reactions = cnn.predict_emotions(content)

    for i, r in enumerate(content):
        print('{}:\n{}'.format(r, predicted_reactions[i]))


if __name__ == '__main__':
    train_and_evaluate_cnn()
