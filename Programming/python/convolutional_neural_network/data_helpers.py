import re

import numpy as np

from importer.database.data_types import Post
from importer.database.mongodb import MongodbStorage


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_training_set(db: MongodbStorage):
    reactions_right_list = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]
    data = []
    reactions_matrix = []

    post_filter = {Post.COLL_MESSAGE: {'$exists': True}}
    for post in db.iterate_single_post(post_filter):
        data.append(post.message)
        reactions_matrix.append(post.reactions)
    [cleaned_posts, new_reactions_matrix] = [[], []]

    for i, post in enumerate(data):
        reactions = reactions_matrix[i]
        reaction_sum = 0
        try:
            if type(reactions) is list:
                integer_reactions = [int(numeric_string) for numeric_string in reactions]
                reaction_sum += sum(integer_reactions)
            else:
                for key in reactions_right_list:
                    reaction_sum += reactions[key]
        except Exception:
            continue
        if np.math.isnan(reaction_sum) or reaction_sum < 5:
            continue

        new_reactions = []
        if type(reactions) is list:
            for reaction in reactions:
                new_reactions.append(int(reaction) / reaction_sum)
        else:
            for key in reactions_right_list:
                new_reactions.append(reactions[key] / reaction_sum)

        cleaned_posts.append(post)
        new_reactions_matrix.append(new_reactions)
    return [cleaned_posts, new_reactions_matrix]
