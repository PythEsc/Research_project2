import csv

from importer.database.data_types import Post, Comment, Emotion
from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage
from nltk.corpus import wordnet as wn


def read_data(database: MongodbStorage, supermarket: str, prefix: str):
    base_string = "../../../../data/Filtered/" + supermarket + "/"
    months = ["March/", "April/", "May/", "June/", "July/", "August/", "September/", "October/", "November/",
              "December/"]
    count = 0
    for month in months:
        # COMMENTS
        f = open(base_string + prefix + "_" + month + "comments.csv", "r")
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            try:
                user_id = row[1].split("_")[0]
                comment_object = Comment.create_from_single_values(comment_id=row[5], parent_id=row[1], content=row[8],
                                                                   date=row[9], user_id=user_id)
                database.insert_comment(comment_object)
            except Exception:
                print("Error while processing post with id: " + user_id)
                count += 1
        # FULLSTATS
        f = open(base_string + prefix + "_" + month + "fullstats.csv", "r")
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            try:
                user_id = row[1].replace("post_user_", "")
                post_object = Post.create_from_single_values(message=row[4], date=row[9], link=row[3], post_id=row[2],
                                                             reactions={'LIKE': row[22], 'LOVE': row[23],
                                                                        'WOW': row[24], 'HAHA': row[25], 'SAD': row[26],
                                                                        'ANGRY': row[27], 'THANKFUL': row[28]},
                                                             user_id=user_id, off_topic=False)
                database.insert_post(post_object)
            except Exception:
                print("Error while processing post with id: " + user_id)
                count += 1

    print("Could not integrate this many posts and comments: " + str(count) + " for " + supermarket)


def learn_more_emotion_words(db: DataStorage):
    counter = 0
    for emotion in db.iterate_single_emotion({}):
        emotion_array = emotion.emotion
        emotion_name = emotion.id
        for ss in wn.synsets(emotion_name):
            last_lemma = emotion_name
            for lemma in ss.lemma_names():
                if lemma == last_lemma:
                    continue
                if lemma != emotion_name and "_" not in lemma:
                    last_lemma = lemma
                    if db.select_single_emotion({Emotion.COLL_ID: lemma}) is None:
                        new_emotion_object = Emotion.create_from_single_values(lemma, emotion_array)
                        db.insert_emotion(new_emotion_object)
                        counter += 1
    return counter


if __name__ == '__main__':
    db = MongodbStorage()
    learn_more_emotion_words(db)
