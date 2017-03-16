import re
from Programming.Importer.DataImporter import DataImporter
import nltk
import string as stringlib
from collections import defaultdict
import numpy as np
import pickle


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            percent = "{0:.1f}".format(100 * (line / float(vocab_size)))
            filled_length = int(100 * line // vocab_size)
            bar = '#' * filled_length + '-' * (100 - filled_length)
            print('\r%s |%s| %s%%' % ('\t', bar, percent), end='')
            word = []
            while True:
                ch = f.read(1).decode("ISO-8859-1")
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    print()
    return word_vecs


def get_w(word_vecs, k=300):
    """
    Get word matrix. w[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    w = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    w[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        w[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return w, word_idx_map


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.lower()
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    # Instead of using those lines we are replacing every "'" with " " such that words like "I'm" -> "I m" and get
    # removed by the latest filter which removes words with length <= 2

    # string = re.sub(r"'s", " 's", string)
    # string = re.sub(r"'ve", " 've", string)
    # string = re.sub(r"n't", " n't", string)
    # string = re.sub(r"'re", " 're", string)
    # string = re.sub(r"'d", " 'd", string)
    # string = re.sub(r"'ll", " 'll", string)

    string = re.sub(r"'", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    # Convert www.* or https?://* to URL
    string = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '__URL__', string)
    # Convert @username to AT_USER
    string = re.sub('@[^\s]+', '__AT_USER__', string)
    # Remove additional white spaces
    string = re.sub('[\s]+', ' ', string)
    # Replace #word with word
    string = re.sub(r'#([^\s]+)', r'\1', string)
    # Replace 3 or more repetitions of character with the character itself
    string = re.sub(r'(.)\1{2,}', r'\1', string)
    # Remove words with 2 or less characters
    string = re.sub(r'\b\w{1,2}\b', '', string)
    # trim
    return string.strip('\'"').strip()


importer_sainsbury = DataImporter("../Filtered/Sainsbury.zip",
                                  "../Unzipped/Sainsbury")

importer_sainsbury.load()

print("---------- Start downloading stopwords ----------")
nltk.download("stopwords")
print("---------- Finished downloading stopwords ----------\n")

punctuation = list(stringlib.punctuation)

stop = nltk.corpus.stopwords.words('english') + punctuation
stop.append('__AT_USER__')
stop.append('__URL__')

print("---------- Start tokenizing data ----------")
vocab = defaultdict(float)
for _, dataMap in importer_sainsbury.data_storage.items():
    for comment_structure in dataMap["comments"]:
        original_comment = comment_structure[8]
        if not isinstance(original_comment, str):
            continue

        cleaned_comment = clean_str(original_comment)
        cleaned_words = cleaned_comment.split()

        tokens = [word for word in cleaned_words if word not in stop]

        for word in set(tokens):
            vocab[word] += 1

print("---------- Finished tokenizing data ----------\n")

print("---------- Start loading pre-trained word2vec model ----------")
dictionary = load_bin_vec("../Model/GoogleNews-vectors-negative300.bin", vocab)
print("---------- Finished loading pre-trained word2vec model ----------\n")

print("---------- Start saving vectors to file ----------")
w, word_idx_map = get_w(dictionary)
pickle.dump([w, word_idx_map, vocab], open("../Model/dataset.p", "wb"))
print("---------- Finished saving vectors to file ----------\n")

print("-------------------- Dataset successfully created --------------------\n")
