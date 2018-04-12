import numpy as np
import re

from importer.database.data_types import Post
from importer.database.database_access import DataStorage


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size)
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


def get_training_set(db: DataStorage, threshold: int = 1, use_likes: bool = False):
    if use_likes:
        reactions_right_list = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]
    else:
        reactions_right_list = ["LOVE", "WOW", "HAHA", "SAD", "ANGRY"]

    data = []
    reactions_matrix = []

    post_filter = {Post.COLL_MESSAGE: {'$exists': True}, Post.COLL_REACTIONS: {'$exists': True}}
    for post in db.iterate_single_post(post_filter):
        data.append(post.message)
        reactions_matrix.append(post.reactions)
    [filtered_posts, new_reactions_matrix] = [[], []]

    for post, reactions in zip(data, reactions_matrix):
        reaction_sum = 0
        for key in reactions_right_list:
            reaction_sum += reactions[key]

        if reaction_sum < threshold:
            continue

        new_reactions = []
        for key in reactions_right_list:
            new_reactions.append(reactions[key] / reaction_sum)

        filtered_posts.append(post)
        new_reactions_matrix.append(new_reactions)

    return [filtered_posts, new_reactions_matrix]


def get_training_set_with_emotions(db: DataStorage, threshold: int = 1, use_likes: bool = False):
    if use_likes:
        reactions_right_list = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]
    else:
        reactions_right_list = ["LOVE", "WOW", "HAHA", "SAD", "ANGRY"]

    data = []
    reactions_matrix = []
    emotions = []

    post_filter = {Post.COLL_MESSAGE: {'$exists': True}, Post.COLL_REACTIONS: {'$exists': True},
                   Post.COLL_COMMENT_EMOTION: {'$exists': True}}
    for post in db.iterate_single_post(post_filter):
        data.append(post.message)
        reactions_matrix.append(post.reactions)
        emotions.append(post.comment_emotion)
    [filtered_posts, new_reactions_matrix, new_emotions] = [[], [], []]

    for post, reactions, emots in zip(data, reactions_matrix, emotions):
        reaction_sum = 0
        for key in reactions_right_list:
            reaction_sum += reactions[key]

        if reaction_sum < threshold:
            continue

        new_reactions = []
        for key in reactions_right_list:
            new_reactions.append(reactions[key] / reaction_sum)

        new_emots = []
        emots = [float(e) for e in emots]
        emots_sum = sum(emots)
        if emots_sum > 1:
            for e in emots:
                new_emots.append(e/emots_sum)
        else:
            new_emots = emots

        filtered_posts.append(post)
        new_reactions_matrix.append(new_reactions)
        new_emotions.append(new_emots)

    return [filtered_posts, new_reactions_matrix, new_emotions]


def clean_text(text):
    clean_text = []
    for t in text:
        # To lower case
        t = t.lower()
        # Add space around punctuation
        t = re.sub('([.\-\"\',:!()$%&\[\]{}?=;#+/*])', r' \1 ', t)
        clean_text.append(t)
    return clean_text
