import codecs
import os
import re

import fasttext
import numpy as np
from tensorflow.contrib import learn

from importer.database.mongodb import MongodbStorage
from neural_networks.util.data_helpers import get_training_set


class WordEmbeddings():
    def __init__(self, dim=50):

        # I only loaded the 50 and 100 dimensions. There are 200 and 300 as well,
        # but since they are too large for our memories, I left them out.
        self.dim = dim

        if dim == 50:
            glovefile = 'glove.6B.50d.txt'
        elif dim == 100:
            glovefile = 'glove.6B.100d.txt'
        else:
            raise ValueError("Dimension '{}' not supported. Try 50 or 100.".format(dim))

        self.glovepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glove', glovefile)
        self.vocab, self.embd = self.load_glove(self.glovepath)

    def load_glove(self, filepath):
        """
        Read GloVe format embedding file.
        :param filepath: The full path to the file.
        :return: vocab, embd. Vocab is a list of the vocabulary. Embd is a numpy array of the embedding, where
                              each row corresponds to a word in vocab.
        """
        print('Loading GloVe vectors from {} ...'.format(filepath))
        vocab = []
        embd = []
        with open(filepath, 'r', encoding='utf8') as file:
            for line in file.readlines():
                row = line.strip().split(' ')
                vocab.append(row[0])
                embd.append(row[1:])
        self.vocab = vocab
        self.embd = np.array(embd, dtype=np.float32)
        self.categorical_vocab = self._create_categorical_vocabulary(self.vocab)

        print('Embedding loaded.')
        print('Vocab size: {}'.format(len(self.vocab)))
        print('Embedding dimension: {}'.format(self.embd.shape[1]))

        return self.vocab, self.embd

    def fit_extra(self, data,
                  min_count=1, word_ngrams=5, minn=3, maxn=6, learning_rate=0.05, n_epochs=100, n_threads=8):
        """
        
        :param data: The list of posts
        :param min_count: The minimum number of occurrences for a word to be considered.
        :param word_ngrams: The maximum word ngram to be considered.
        :param minn: The minimum character ngram.
        :param maxn: The maximum character ngram.
        :param learning_rate: The learning rate of the algorithm.
        :param n_epochs: The number of epochs to train on.
        :param n_threads: The number of cores to use, the more the faster.
        :return: 
        """

        # need these paths because fasttext reads from file
        text_file = './text.txt'
        output_model = './embedding'

        data = self._clean_text(data)
        self._save_input(data, text_file)

        # train the model
        print('\nFitting extra data...')
        fasttext.skipgram(input_file=text_file,
                          output=output_model,
                          dim=self.dim,
                          min_count=min_count,
                          word_ngrams=word_ngrams,
                          minn=minn,
                          maxn=maxn,
                          lr=learning_rate,
                          epoch=n_epochs,
                          thread=n_threads,
                          silent=False)
        print('Finished fitting.\n')

        self._save_output(self.glovepath, ''.join([output_model, '.vec']))

        os.remove(text_file)
        os.remove(''.join([output_model, '.vec']))
        os.remove(''.join([output_model, '.bin']))

        self.vocab, self.embd = self.load_glove(self.glovepath)

    def _clean_text(self, text):
        clean_text = []
        for t in text:
            # To lower case
            t = t.lower()
            # Add space around punctuation
            t = re.sub('([.\-\"\',:!()$%&\[\]{}?=;#+/*])', r' \1 ', t)
            clean_text.append(t)
        return clean_text

    def _save_output(self, glove_file, fasttext_file):
        fasttext_embedding = self._load_fasttext(fasttext_file)

        words_to_save = set(fasttext_embedding.keys()) - set(self.vocab)
        print('Saving {} new words to {} ...'.format(len(words_to_save), glove_file))

        with codecs.open(glove_file, 'a', encoding='utf-8') as file:
            for i, word in enumerate(words_to_save):
                print("\r%.2f%%" % (i / len(words_to_save) * 100), end='')
                file.write('{} {}\n'.format(word, ' '.join(fasttext_embedding[word])))

    @staticmethod
    def _load_fasttext(filepath):
        fasttext_embedding = {}
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file.readlines()):
                if i > 0:
                    # The expected format of each line in  the file is:
                    # <word><space>(<double><space>)+
                    row = line.strip().split(' ')
                    fasttext_embedding[row[0]] = row[1:]

        return fasttext_embedding

    @staticmethod
    def _save_input(X, fpath):
        with codecs.open(fpath, 'w', 'utf-8') as file:
            for x in X:
                file.write('{}\n'.format(re.sub('\n|\r\n|\r', ' ', x)))

    @staticmethod
    def _create_categorical_vocabulary(words):
        vocab = learn.preprocessing.CategoricalVocabulary()
        for w in words:
            vocab.add(w)
        return vocab


if __name__ == '__main__':
    print('Loading training data...')
    db = MongodbStorage()
    x_text, y = get_training_set(db)

    we = WordEmbeddings(50)
    we.fit_extra(x_text)
