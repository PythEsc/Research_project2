import re

from importer.database.data_types import Post
from importer.database.database_access import DataStorage


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

        if len(filtered_posts) >= 50000:
            break

    print("Finished loading dataset containing {} samples".format(len(filtered_posts)))

    return [filtered_posts, new_reactions_matrix]


def training_set_iter(db: DataStorage, threshold: int = 1, use_likes: bool = False):
    if use_likes:
        reactions_right_list = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]
    else:
        reactions_right_list = ["LOVE", "WOW", "HAHA", "SAD", "ANGRY"]

    post_filter = {Post.COLL_MESSAGE: {'$exists': True}, Post.COLL_REACTIONS: {'$exists': True}}
    for post in db.iterate_single_post(post_filter):
        message = post.message
        reactions = post.reactions

        reaction_sum = 0
        for key in reactions_right_list:
            reaction_sum += reactions[key]

        if reaction_sum < threshold:
            continue

        new_reactions = []
        for key in reactions_right_list:
            new_reactions.append(reactions[key] / reaction_sum)

        message = clean_text(text=[message])[0]

        yield message, new_reactions


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
                new_emots.append(e / emots_sum)
        else:
            new_emots = emots

        filtered_posts.append(post)
        new_reactions_matrix.append(new_reactions)
        new_emotions.append(new_emots)

    return [filtered_posts, new_reactions_matrix, new_emotions]


def clean_text(text: list):
    clean_text = []
    for t in text:
        # To lower case
        t = t.lower()
        # Add space around punctuation
        t = re.sub('([.\-\"\',:!()$%&\[\]{}?=;#+/*])', r' \1 ', t)
        clean_text.append(t)
    return clean_text
