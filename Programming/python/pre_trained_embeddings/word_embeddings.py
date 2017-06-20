import os
import numpy as np
from tensorflow.contrib import learn


class WordEmbeddings():
    def __init__(self, dim=50):

        # I only loaded the 50 and 100 dimensions. There are 200 and 300 as well,
        # but since they are too large for our memories, I left them out.
        if dim not in [50, 100]:
            raise ValueError("Dimension not supported. Try 50 or 100.")
        self.dim = dim

        if dim == 50:
            file_name = 'glove.6B.50d.txt'
        elif dim == 100:
            file_name = 'glove.6B.100d.txt'

        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glove', file_name)

        self.vocab, self.embd = self.loadGloVe(file_path)

        self.categorical_vocab = self.createCategoricalVocabulary(self.vocab)

        # print('Embedding loaded.')
        # print('Vocab size: {}'.format(len(self.vocab)))
        # print('Embedding dimension: {}'.format(self.embd.shape[1]))

    def createCategoricalVocabulary(self, words):
        vocab = learn.preprocessing.CategoricalVocabulary()
        for w in words:
            vocab.add(w)
        return vocab

    def loadGloVe(self, filename):
        """
        Read GloVe format embedding file.
        :param filename: The full path to the file.
        :return: vocab, embd. Vocab is a list of the vocabulary. Embd is a numpy array of the embedding, where
                              each row corresponds to a word in vocab.
        """
        vocab = []
        embd = []
        file = open(filename, 'r', encoding='utf8')
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
        file.close()
        return vocab, np.array(embd).astype(np.float32)


if __name__ == '__main__':
    WordEmbeddings(50)
