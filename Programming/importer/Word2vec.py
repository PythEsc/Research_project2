import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pickle
import string as stringlib
from collections import defaultdict
from Programming.importer.DataImporter import DataImporter
from Programming.importer.word2vec_utitlity import clean_str
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

def plot_with_labels(low_dim_embs, labels, filename='../Model/dataset.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(25, 25))  # in inches
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

    try:
        print("---------- Start plotting data ----------")
        # Use t-distributed stochastic neighbor embedding for displaying the data
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

        # The number of words you want to add to the graph
        plot_only = 500
        # Loading data into graph
        low_dim_embs = tsne.fit_transform(w[1:plot_only, :])

        # Create labels for the added words
        labels = np.empty(plot_only - 1, dtype=object)
        for i in range(1, plot_only):
            labels[i - 1] = idx_word_map[i]

        plot_with_labels(low_dim_embs, labels, filename="../Model/sainsbury.png")
        print("---------- Finished plotting data ----------")
    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
        raise