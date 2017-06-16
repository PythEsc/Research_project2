#! /usr/bin/env python

import csv
import os

import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.contrib import learn

from importer.data_importer import DataImporter
# Data Parameters
# Parameters
# ==================================================
from neural_networks import data_helpers

tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1495884681/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
importer_sainsbury = DataImporter("../../../data/Filtered/Sainsbury.zip", "../../../data/Unzipped/Sainsbury")
importer_sainsbury.load()
x_text, y = importer_sainsbury.prepare_data_for_CNN()
# if FLAGS.eval_train:
# x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
# y_test = np.argmax(y_test, axis=1)
# else:
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = np.copy(x_text)
y_shuffled = np.copy(y)
for i in range(len(shuffle_indices)):
    value = y[shuffle_indices[i]]
    y_shuffled[i] = value

    value = x_text[shuffle_indices[i]]
    x_shuffled[i] = value

x_raw = x_shuffled[:20]
y_test = y_shuffled[:20]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
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

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1})
            if len(all_predictions) == 0:
                all_predictions = batch_predictions
            else:
                all_predictions = np.vstack([all_predictions, batch_predictions])

# Save the evaluation to a csv
mae = metrics.mean_absolute_error(y_test, all_predictions)
mse = metrics.mean_squared_error(y_test, all_predictions)
print('\nMean absolute error: {}'.format(mae))
print('Mean squared error: {}'.format(mse))
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
