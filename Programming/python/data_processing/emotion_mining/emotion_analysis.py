import numpy as np
import pandas

from importer.database.data_types import Post, Emotion
from importer.database.mongodb import MongodbStorage

THRESHOLD = 5

reaction_order = ['LOVE', 'WOW', 'HAHA', 'SAD', 'ANGRY']


def main():
    correlation_matrix = np.zeros((len(reaction_order), len(Emotion.EMOTION_TYPES)))
    correlation_matrix_count = np.zeros((len(reaction_order), len(Emotion.EMOTION_TYPES)))

    db = MongodbStorage()
    counter = 0
    for post in db.iterate_single_post({Post.COLL_COMMENT_EMOTION: {"$exists": True}}):
        counter += 1
        mined_emotions = post.comment_emotion
        real_emotions = [-1] * len(reaction_order)
        for reaction, amount in post.reactions.items():
            found_index = -1
            for index, key in enumerate(reaction_order):
                if key == reaction:
                    found_index = index
                    break
            if found_index != -1:
                real_emotions[found_index] = amount

        sum_real_emotions = sum(real_emotions)
        if sum_real_emotions < THRESHOLD:
            continue

        sum_mined_emotions = sum(mined_emotions)
        # if sum_mined_emotions == 0:
        #     continue

        ratio_mined_emotions = [mined_emotion / sum_mined_emotions if sum_mined_emotions != 0 else 0 for mined_emotion
                                in mined_emotions]
        ratio_real_emotions = [real_emotion / sum_real_emotions if sum_real_emotions != 0 else 0 for real_emotion in
                               real_emotions]

        for index_real, real_emotion in enumerate(ratio_real_emotions):
            for index_mined, mined_emotion in enumerate(ratio_mined_emotions):
                diff = abs(real_emotion - mined_emotion)
                if diff != 0:
                    correlation_matrix_count[index_real][index_mined] += 1
                correlation_matrix[index_real][index_mined] += diff

    correlation_matrix /= correlation_matrix_count

    for array in correlation_matrix:
        for index, entry in enumerate(array):
            array[index] = round(entry, 3)

    row_labels = reaction_order
    column_labels = Emotion.EMOTION_TYPES

    df = pandas.DataFrame(correlation_matrix, columns=column_labels, index=row_labels)
    print(str(df))

    df = pandas.DataFrame(correlation_matrix_count, columns=column_labels, index=row_labels)
    print(str(df))


if __name__ == '__main__':
    main()
