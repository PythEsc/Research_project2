#!/usr/bin/python3

#
#  Copyright 2016 Peter de Vocht
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import string
from typing import List

import gensim
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
# an embedding word with associated vector
from sklearn.metrics.pairwise import cosine_similarity


class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector


# a sentence, a list of words
class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    # return the length of a sentence
    def len(self) -> int:
        return len(self.word_list)


# todo: get the frequency for a word in a document set
def get_word_frequency(word_text):
    return 1.0


# A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS
# Sanjeev Arora, Yingyu Liang, Tengyu Ma
# Princeton University
# convert a list of sentence with word2vec items into a set of sentence vectors
def sentence_to_vec(sentence_list: List[Sentence], embedding_size, a=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        for word in sentence.word_list:
            a_value = a / (a + get_word_frequency(word.text))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word.vector))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences

    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, vs = vs -u x uT x vs
    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))

    return sentence_vecs


def avg_vector(word_embeddings, words):
    sum_of_words = [0] * 300
    for word in word_embeddings:
        sum_of_words = [x + y for x, y in zip(word.vector, sum_of_words)]
    sum_of_words = [x / len(words) for x in sum_of_words]
    return sum_of_words


# test
embedding_size = 300  # dimension of the word embedding

# some random word/GloVe vectors for demo purposes of dimension 300
sentence_str1 = "I really love your car."
sentence_str2 = "I really hate your car."
words1 = nltk.word_tokenize(sentence_str1)
words2 = nltk.word_tokenize(sentence_str2)

stop = set(stopwords.words('english'))
punctuation = string.punctuation

words1 = [word.lower() for word in words1 if word not in stop and word not in punctuation]
words2 = [word.lower() for word in words2 if word not in stop and word not in punctuation]

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/florian/Workspace/text-mining/data/GoogleNews-vectors-negative300.bin', binary=True)

word_embeddings1 = [Word(word, w2v_model[word]) for word in words1]
word_embeddings2 = [Word(word, w2v_model[word]) for word in words2]

avg1 = avg_vector(word_embeddings1, words1)
avg2 = avg_vector(word_embeddings2, words2)

# create some artificial sentences using these words
sentence1 = Sentence(word_embeddings1)
sentence2 = Sentence(word_embeddings2)

# calculate and display the result
sentence_vectors = sentence_to_vec([sentence1, sentence2], embedding_size)
print("Sentence2Vec: " + str(cosine_similarity(sentence_vectors[0], sentence_vectors[1])))
print("AvgVec: " + str(cosine_similarity(avg1, avg2)))
