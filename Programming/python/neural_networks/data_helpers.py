import numpy as np

from importer.database.data_types import Post
from importer.database.database_access import DataStorage


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) +1
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
