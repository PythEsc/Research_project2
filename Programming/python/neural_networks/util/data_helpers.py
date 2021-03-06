import re
from datetime import datetime
from typing import Optional

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


def training_set_iter(db: DataStorage, threshold: int = 1, use_likes: bool = False, allow_augmented_data: bool = True,
                      max_post_length: Optional[int] = None):
    if use_likes:
        reactions_right_list = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]
    else:
        reactions_right_list = ["LOVE", "WOW", "HAHA", "SAD", "ANGRY"]

    skipped_counter_reactions = 0
    skipped_counter_length = 0

    post_filter = {Post.COLL_MESSAGE: {'$exists': True}, Post.COLL_REACTIONS: {'$exists': True},
                   Post.COLL_DATE: {"$gte": datetime(year=2016, month=12, day=31)}}
    count_with_augmented = db.count_posts(filter=post_filter)
    if not allow_augmented_data:
        post_filter[Post.COLL_AUGMENTED] = False

    counter = 0
    for post in db.iterate_single_post(post_filter):
        message = post.message
        reactions = post.reactions

        reaction_sum = 0
        for key in reactions_right_list:
            reaction_sum += reactions[key]

        if reaction_sum < threshold:
            skipped_counter_reactions += 1
            continue

        new_reactions = []
        for key in reactions_right_list:
            new_reactions.append(reactions[key] / reaction_sum)

        message = clean_text(text=[message])[0]

        if max_post_length is not None:
            length = len(message.split(" "))
            if length > max_post_length:
                skipped_counter_length += 1
                continue

        counter += 1
        yield message, new_reactions

    if threshold > 0:
        print("{} posts were skipped because they did not fulfill the reaction-sum requirement".format(
            str(skipped_counter_reactions)))
    if max_post_length is not None:
        print("{} posts were skipped because they did not fulfill the maximum-length requirement".format(
            str(skipped_counter_length)))
    if not allow_augmented_data:
        print("{} posts were skipped because they did not fulfill the not-augmented requirement".format(
            str(count_with_augmented - counter)))


def validation_set_iter(db: DataStorage, threshold: int = 1, use_likes: bool = False, allow_augmented_data: bool = True,
                        max_post_length: Optional[int] = None):
    if use_likes:
        reactions_right_list = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]
    else:
        reactions_right_list = ["LOVE", "WOW", "HAHA", "SAD", "ANGRY"]

    skipped_counter_reactions = 0
    skipped_counter_length = 0

    post_filter = {Post.COLL_MESSAGE: {'$exists': True}, Post.COLL_REACTIONS: {'$exists': True},
                   Post.COLL_DATE: {"$lt": datetime(year=2016, month=12, day=31)}}
    count_with_augmented = db.count_posts(filter=post_filter)
    if not allow_augmented_data:
        post_filter[Post.COLL_AUGMENTED] = False

    counter = 0
    for post in db.iterate_single_post(post_filter):
        message = post.message
        reactions = post.reactions

        reaction_sum = 0
        for key in reactions_right_list:
            reaction_sum += reactions[key]

        if reaction_sum < threshold:
            skipped_counter_reactions += 1
            continue

        new_reactions = []
        for key in reactions_right_list:
            new_reactions.append(reactions[key] / reaction_sum)

        message = clean_text(text=[message])[0]

        if max_post_length is not None:
            length = len(message.split(" "))
            if length > max_post_length:
                skipped_counter_length += 1
                continue

        counter += 1
        yield message, new_reactions

    if threshold > 0:
        print("{} posts were skipped because they did not fulfill the reaction-sum requirement".format(
            str(skipped_counter_reactions)))
    if max_post_length is not None:
        print("{} posts were skipped because they did not fulfill the maximum-length requirement".format(
            str(skipped_counter_length)))
    if not allow_augmented_data:
        print("{} posts were skipped because they did not fulfill the not-augmented requirement".format(
            str(count_with_augmented - counter)))


def training_set_with_emotions_iter(db: DataStorage, threshold: int = 1, use_likes: bool = False,
                                    allow_augmented_data: bool = True,
                                    max_post_length: Optional[int] = None):
    if use_likes:
        reactions_right_list = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]
    else:
        reactions_right_list = ["LOVE", "WOW", "HAHA", "SAD", "ANGRY"]

    skipped_counter_reactions = 0
    skipped_counter_length = 0

    post_filter = {Post.COLL_MESSAGE: {'$exists': True},
                   Post.COLL_REACTIONS: {'$exists': True},
                   Post.COLL_COMMENT_EMOTION: {'$exists': True},
                   Post.COLL_DATE: {"$gte": datetime(year=2016, month=12, day=31)}}

    count_with_augmented = db.count_posts(filter=post_filter)
    if not allow_augmented_data:
        post_filter[Post.COLL_AUGMENTED] = False

    counter = 0
    for post in db.iterate_single_post(post_filter):
        message = post.message
        reactions = post.reactions
        emotions = post.comment_emotion

        reaction_sum = 0
        for key in reactions_right_list:
            reaction_sum += reactions[key]

        if reaction_sum < threshold:
            skipped_counter_reactions += 1
            continue

        new_reactions = []
        for key in reactions_right_list:
            new_reactions.append(reactions[key] / reaction_sum)

        new_emots = []
        emots = [float(e) for e in emotions]
        emots_sum = sum(emots)
        if emots_sum > 1:
            for e in emots:
                new_emots.append(e / emots_sum)
        else:
            new_emots = emots

        message = clean_text(text=[message])[0]

        if max_post_length is not None:
            length = len(message.split(" "))
            if length > max_post_length:
                skipped_counter_length += 1
                continue

        counter += 1
        yield message, new_reactions, new_emots

    if threshold > 0:
        print("{} posts were skipped because they did not fulfill the reaction-sum requirement".format(
            str(skipped_counter_reactions)))
    if max_post_length is not None:
        print("{} posts were skipped because they did not fulfill the maximum-length requirement".format(
            str(skipped_counter_length)))
    if not allow_augmented_data:
        print("{} posts were skipped because they did not fulfill the not-augmented requirement".format(
            str(count_with_augmented - counter)))


def validation_set_with_emotions_iter(db: DataStorage, threshold: int = 1, use_likes: bool = False,
                                      allow_augmented_data: bool = True,
                                      max_post_length: Optional[int] = None):
    if use_likes:
        reactions_right_list = ["LIKE", "LOVE", "WOW", "HAHA", "SAD", "ANGRY"]
    else:
        reactions_right_list = ["LOVE", "WOW", "HAHA", "SAD", "ANGRY"]

    skipped_counter_reactions = 0
    skipped_counter_length = 0

    post_filter = {Post.COLL_MESSAGE: {'$exists': True},
                   Post.COLL_REACTIONS: {'$exists': True},
                   Post.COLL_COMMENT_EMOTION: {'$exists': True},
                   Post.COLL_DATE: {"$lt": datetime(year=2016, month=12, day=31)}}

    count_with_augmented = db.count_posts(filter=post_filter)
    if not allow_augmented_data:
        post_filter[Post.COLL_AUGMENTED] = False

    counter = 0
    for post in db.iterate_single_post(post_filter):
        message = post.message
        reactions = post.reactions
        emotions = post.comment_emotion

        reaction_sum = 0
        for key in reactions_right_list:
            reaction_sum += reactions[key]

        if reaction_sum < threshold:
            skipped_counter_reactions += 1
            continue

        new_reactions = []
        for key in reactions_right_list:
            new_reactions.append(reactions[key] / reaction_sum)

        new_emots = []
        emots = [float(e) for e in emotions]
        emots_sum = sum(emots)
        if emots_sum > 1:
            for e in emots:
                new_emots.append(e / emots_sum)
        else:
            new_emots = emots

        message = clean_text(text=[message])[0]

        if max_post_length is not None:
            length = len(message.split(" "))
            if length > max_post_length:
                skipped_counter_length += 1
                continue

        counter += 1
        yield message, new_reactions, new_emots

    if threshold > 0:
        print("{} posts were skipped because they did not fulfill the reaction-sum requirement".format(
            str(skipped_counter_reactions)))
    if max_post_length is not None:
        print("{} posts were skipped because they did not fulfill the maximum-length requirement".format(
            str(skipped_counter_length)))
    if not allow_augmented_data:
        print("{} posts were skipped because they did not fulfill the not-augmented requirement".format(
            str(count_with_augmented - counter)))


def clean_text(text: list):
    clean_text = []
    for t in text:
        # To lower case
        t = t.lower()
        # Add space around punctuation
        t = re.sub('([.\-\"\',:!()$%&\[\]{}?=;#+/*])', r' \1 ', t)
        clean_text.append(t)
    return clean_text
