import tensorflow as tf


def save_model(saver: tf.train.Saver, sess: tf.Session, checkpoint_dir: str):
    """
    Save a model that was initialized with the provided Session.

    :param saver: The saver object to be used during saving.
    :param sess:  The session to be saved.
    :param checkpoint_dir: The directory to save the model.
    :return: 
    """
    print("Saving model to {}\n".format(str(checkpoint_dir)))
    saver.save(sess, checkpoint_dir)


def restore_model(sess: tf.Session, checkpoint_dir: str) -> tf.Session:
    """
    Loads the latest model from checkpoint_dir.

    :param sess: The Session where the model will be loaded into.
    :param checkpoint_dir: The directory of the saved model.
    :return: The provided Session with loaded parameters.
    """
    print("Restoring model from {}\n".format(str(checkpoint_dir)))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    return sess
