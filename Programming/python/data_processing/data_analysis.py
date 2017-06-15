from importer.data_retrieval.facebook import facebook_parser
from importer.database.mongodb import MongodbStorage

THRESHOLDS = [0, 1, 2, 5, 10]


def main():
    post_filtered_with_likes = {}
    post_filtered_without_likes = {}
    post_filtered_without_likes_including_each_post_once = {}

    for threshold in THRESHOLDS:
        post_filtered_with_likes[threshold] = []
        post_filtered_without_likes[threshold] = []
        post_filtered_without_likes_including_each_post_once[threshold] = []

    # Iterate over posts and sort them into the different threshold lists
    db = MongodbStorage()
    for post in db.iterate_single_post({}):
        post_reactions_with_likes = 0
        post_reactions_without_likes = 0
        for reaction, amount in post.reactions.items():
            if reaction != 'LIKE':
                post_reactions_without_likes += amount
            post_reactions_with_likes += amount

        highest_threshold = 0
        for threshold in THRESHOLDS:
            if post_reactions_with_likes >= threshold:
                post_filtered_with_likes[threshold].append(post)
            if post_reactions_without_likes >= threshold:
                post_filtered_without_likes[threshold].append(post)
                highest_threshold = threshold

        post_filtered_without_likes_including_each_post_once[highest_threshold].append(post)

    # print the total number of posts for each threshold
    print("\n---------- INCLUDING LIKES ----------\n")
    [print("%d: %d" % (threshold, len(posts))) for threshold, posts in post_filtered_with_likes.items()]
    print("\n---------- EXCLUDING LIKES ----------\n")
    [print("%d: %d" % (threshold, len(posts))) for threshold, posts in post_filtered_without_likes.items()]

    # Get the overall distribution of the reactions used for the different thresholds
    for threshold, posts in post_filtered_without_likes.items():

        if threshold == 0:
            continue

        reactions = {}
        for reaction in facebook_parser.FacebookParser.CONST_REACTIONS_TYPES[1:]:
            reactions[reaction] = 0

        for post in posts:
            for reaction, value in post.reactions.items():
                if reaction == 'LIKE':
                    continue
                reactions[reaction] += value

        print("Threshold: %d, Reaction distribution: %s" % (threshold, str(reactions)))

    # Get the ratio on how much people agree to each other with picking their reactions
    for threshold, posts in post_filtered_without_likes_including_each_post_once.items():

        if threshold == 0:
            continue

        used_reaction_list = [0] * 5

        for post in posts:
            reaction_amount = []
            for reaction, value in post.reactions.items():
                if reaction == 'LIKE':
                    continue
                reaction_amount.append(value)

            reaction_amount.sort(reverse=True)
            total_reactions = sum(reaction_amount)

            reaction_ratio = [x / total_reactions for x in reaction_amount]

            used_reaction_list = [x + y for x, y in zip(used_reaction_list, reaction_ratio)]

        used_reaction_list = [x / len(posts) for x in used_reaction_list]
        print("Threshold: %d, User opinion matching: %s" % (threshold, str(used_reaction_list)))


if __name__ == '__main__':
    main()
