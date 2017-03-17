import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pickle
import re
import string as stringlib
from collections import defaultdict
from Programming.Importer.DataImporter import DataImporter
from sklearn.manifold import TSNE


def load_bin_vec(filename, vocabulary):
    word_vecs = {}
    # Open the file <filename> as binary
    with open(filename, "rb") as f:
        # Read the first line that contains the header
        header = f.readline()
        # Split the header and get the size information
        vocab_size, layer1_size = map(int, header.split())
        # The length of <layer1_size> 32bit floats
        binary_len = np.dtype('float32').itemsize * layer1_size

        for line in range(vocab_size):
            # Create an interactive loading bar that displays the current status of processing the file
            percent = "{0:.1f}".format(100 * (line / float(vocab_size)))
            filled_length = int(100 * line // vocab_size)
            bar = '#' * filled_length + '-' * (100 - filled_length)
            print('\r%s |%s| %s%%' % ('\t', bar, percent), end='')

            word = []
            # Read bytes as long as there is no space
            while True:
                # The binary file is encoded as ISO-8859-1
                ch = f.read(1).decode("ISO-8859-1")
                # If we read a blank character the word is finished -> We join all the characters to a string and break
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            # If the word is in our vocabulary we will add it to our word_vector
            if word in vocabulary:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    print()
    return word_vecs


def get_w(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    idx_word_map = dict()

    w = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    w[0] = np.zeros(k, dtype='float32')

    i = 1
    # Iterate over the keys (so the words) of our vocabulary
    for word in word_vecs:
        # Add each word's numerical representation to w
        w[i] = word_vecs[word]
        # Add the entry to the maps that helps us finding entries in w on later usage
        word_idx_map[word] = i
        idx_word_map[i] = word

        i += 1
    return w, word_idx_map, idx_word_map


def clean_str(string):
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
    # Convert www.* or https?://* to __URL__
    string = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '__URL__', string)
    # Convert @username to __AT_USER__
    string = re.sub('@[^\s]+', '__AT_USER__', string)
    # Remove additional white spaces
    string = re.sub('[\s]+', ' ', string)
    # Replace #word with word
    string = re.sub(r'#([^\s]+)', r'\1', string)
    # Replace 3 or more repetitions of character with the character itself
    string = re.sub(r'(.)\1{2,}', r'\1', string)
    # Remove words with 2 or less characters
    string = re.sub(r'\b\w{1,2}\b', '', string)
    # Remove sequences that contain numbers
    string = re.sub(r'\b\w*\d\w*\b', '', string)
    # trim
    return string.strip('\'"').strip()


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

# Create our DataImporter for the Sainsbury data set
importer_sainsbury = DataImporter("../Filtered/Sainsbury.zip",
                                  "../Unzipped/Sainsbury")
# Load the data
importer_sainsbury.load()

# If the vectorised data set already exists and could be loaded from file ask user if he wants to load it or
# overwrite it
if os.path.exists("../Model/dataset.p"):
    input_string = input("There is an already existing vectorised data set. Do you want to load (l) it or overwrite ("
                         "o) it?")

    while input_string != "l" and input_string != "o":
        input_string = input("Please either overwrite the existing file by writing 'o' or load it by writing 'l'")

    if input_string == "o":
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
        w, word_idx_map, idx_word_map = get_w(dictionary)
        pickle.dump([w, word_idx_map, idx_word_map, vocab], open("../Model/dataset.p", "wb"))
        print("---------- Finished saving vectors to file ----------\n")

        print("-------------------- Dataset successfully created --------------------\n")
    else:
        w, word_idx_map, idx_word_map, vocab = pickle.load(open("../Model/dataset.p", "rb"))
        print("\n---------- File successfully loaded ----------")
        print("\tLoaded variables: w, word_idx_map, vocab")

    # try:
    #     print("---------- Start plotting data ----------")
    #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    #     plot_only = 500
    #     low_dim_embs = tsne.fit_transform(w[:plot_only, :])
    #
    #     labels = []
    #     for vec in w[:plot_only]:
    #         print()
    #     labels = [idx_word_map[i] for i in w[:plot_only]]
    #     plot_with_labels(low_dim_embs, labels)
    #     print("---------- Finished plotting data ----------")
    # except ImportError:
    #     print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
    #     raise

