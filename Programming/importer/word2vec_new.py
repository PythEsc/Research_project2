from gensim.models import Word2Vec
from Programming.importer.training_data_iterator import LineIterator
from Programming.importer.training_data_generator import WikipediaPageProcessor
from Programming.importer.DataImporter import DataImporter
from tensorflow.contrib.tensorboard.plugins import projector
import logging
import os
import numpy as np
import Programming.importer.word2vec_utitlity as util
import tensorflow as tf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
training_data_directory = "../training/"
data_set_dir = training_data_directory+"model/"
data_set_name = data_set_dir+"new_dataset"


def preprocessing_filter(comment, stop):
    cleaned_comment = util.clean_str(comment)
    cleaned_words = cleaned_comment.split()

    tokens = [word for word in cleaned_words if word not in stop]
    return tokens

if os.path.exists(data_set_name):
    model = Word2Vec.load(data_set_name)
else:
    titles = ["Sainsbury's", "Tesco", "Supermarket", "Customer", "Sales", "Woman", "Man", "Chess", "Subway (restaurant)"]
    wiki_page_processor = WikipediaPageProcessor(titles, training_data_directory=training_data_directory)
    wiki_page_processor.download()
    wiki_page_processor.process()

    # Create our DataImporter for the Sainsbury data set
    importer_sainsbury = DataImporter("../Filtered/Sainsbury.zip", "../Unzipped/Sainsbury")
    importer_sainsbury.load()

    stopwords = util.get_stopwords()

    model = Word2Vec(min_count=2, size=300)
    model.build_vocab(importer_sainsbury.iterate(lambda comment: preprocessing_filter(comment, stop=stopwords)))
    model.intersect_word2vec_format(fname="../Model/GoogleNews-vectors-negative300.bin.gz", binary=True)

    lines = LineIterator(dirname="../training/processed")
    model.train(lines)

    os.makedirs(os.path.dirname(data_set_name), exist_ok=True)
    model.save(data_set_name)

log_dir = "../tensorboard/"
os.makedirs(os.path.dirname(log_dir), exist_ok=True)

# project part of vocab, 10K of 300 dimension
w2v_10K = np.zeros((1000, 300))
with open(os.path.join(log_dir, 'word_metadata.tsv'), 'w+') as file_metadata:
    for i, word in enumerate(model.wv.index2word[:1000]):
        w2v_10K[i] = model[word]
        file_metadata.write(word + os.linesep)

sess = tf.InteractiveSession()

with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v_10K, trainable=False, name='word_embedding')

tf.global_variables_initializer().run()

saver = tf.train.Saver()
saver.save(sess, os.path.join(log_dir, 'model.ckpt'), global_step=1000)

# adding into projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = os.path.abspath(os.path.join(log_dir, 'word_metadata.tsv'))

tf.global_variables_initializer().run()

writer = tf.summary.FileWriter(log_dir)

# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)

# open tensorboard with logdir, check localhost:6006 for viewing your embedding.
# tensorboard --logdir="../tensorboard/"
