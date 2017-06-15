import os
import traceback

from pycorenlp.corenlp import StanfordCoreNLP

from importer.database.data_types import Emotion, Comment, Post
from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage


class EmotionMiner:
    """
    Class that iterates over a given DataStorage implementation (e.g. MongodbStorage) and processes all entries that do not have a NER tag
    """

    def __init__(self, data_storage: DataStorage):
        """
        Const
        :param data_storage: The DataStorage implementation that contains the data which will be tagged
        """

        self.data_storage = data_storage
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.properties_pos = {'annotators': 'lemma', 'outputFormat': 'json'}

    def get_post_emotion_value(self, content: str):
        emotion_count_list = [0, 0, 0, 0, 0, 0, 0, 0]
        output = self.nlp.annotate(content, properties=self.properties_pos)
        for sent in output["sentences"]:
            for token in sent["tokens"]:
                emotion_filter = {Emotion.COLL_ID: token["lemma"]}
                emotion_object = self.data_storage.select_single_emotion(emotion_filter)
                if emotion_object is not None:
                    emotion_count_list = [sum(x) for x in zip(emotion_count_list, emotion_object.emotion)]
        return emotion_count_list

    def tag_comments_emotions_to_post(self):
        """
        Iterates over all entries in the given DataStorage that do not have a NER tag yet and creates those
        :return: -
        """
        filter = {Post.COLL_COMMENT_EMOTION: {'$exists': False}}
        for post in self.data_storage.iterate_single_post(filter):
            try:
                filter_comment = {Comment.COLL_PARENT_ID: post.post_id}

                comment_collection = ''
                for comment in self.data_storage.iterate_single_comment(filter_comment, False):
                    comment_collection = comment_collection + os.linesep + comment.content

                # Here we need to mine the emotion of all words in the comment.
                # So we need to lookup the words in the dictionary.
                post.comment_emotion = self.get_post_emotion_value(comment_collection)
                self.data_storage.update_post(post)
            except Exception:
                print("Error while processing post with id: " + post.post_id)
                traceback.print_exc()

    def tag_posts_emotions(self):
        """
        Iterates over all entries in the given DataStorage that do not have a NER tag yet and creates those
        :return: -
        """
        filter = {Post.COLL_EMOTION: {'$exists': False}}
        for post in self.data_storage.iterate_single_post(filter):
            try:
                message = post.message
                post.emotion = self.get_post_emotion_value(message)
                self.data_storage.update_post(post)
            except Exception:
                print("Error while processing post with id: " + post.post_id)
                traceback.print_exc()


if __name__ == '__main__':
    db = MongodbStorage()
    """
    tagger = EmotionMiner(data_storage=db)
    tagger.tag_posts_emotions()
    tagger.tag_comments_emotions_to_post()
    """
    filter = {Post.COLL_LINK: {"$regex": "https://www.facebook.com/t.+"}}
    for post in db.iterate_single_post(filter):
        print("\n" + post.message + "\n")
        # Get the emotions
        total_emotions = post.comment_emotion
        sum_emotions = 0
        for emotion in total_emotions:
            sum_emotions += emotion

        chosen_emotions = []
        if sum_emotions > 0:
            relative_emotions = []
            for emotion in total_emotions:
                relative_emotions.append(emotion / sum_emotions)

            total_emotions_to_display = 0.4

            while total_emotions_to_display > 0:
                maximum_emotion_value = 0
                maximum_emotion_index = -1

                for index, emotion in enumerate(relative_emotions):
                    if emotion > maximum_emotion_value and (index, emotion) not in chosen_emotions:
                        maximum_emotion_index = index
                        maximum_emotion_value = emotion
                chosen_emotions.append((maximum_emotion_index, maximum_emotion_value))
                total_emotions_to_display -= maximum_emotion_value

        displayed_emotions = []
        for emotion_tuple in chosen_emotions:
            displayed_emotions.append(
                "{emotion_name}: {emotion_value:.1f}%".format(emotion_name=Emotion.EMOTION_TYPES[emotion_tuple[0]],
                                                              emotion_value=emotion_tuple[1] * 100))

        print(displayed_emotions)

        total_emotions = post.emotion
        sum_emotions = 0
        for emotion in total_emotions:
            sum_emotions += emotion

        chosen_emotions = []
        if sum_emotions > 0:
            relative_emotions = []
            for emotion in total_emotions:
                relative_emotions.append(emotion / sum_emotions)

            total_emotions_to_display = 0.4

            while total_emotions_to_display > 0:
                maximum_emotion_value = 0
                maximum_emotion_index = -1

                for index, emotion in enumerate(relative_emotions):
                    if emotion > maximum_emotion_value and (index, emotion) not in chosen_emotions:
                        maximum_emotion_index = index
                        maximum_emotion_value = emotion
                chosen_emotions.append((maximum_emotion_index, maximum_emotion_value))
                total_emotions_to_display -= maximum_emotion_value

        displayed_emotions = []
        for emotion_tuple in chosen_emotions:
            displayed_emotions.append(
                "{emotion_name}: {emotion_value:.1f}%".format(emotion_name=Emotion.EMOTION_TYPES[emotion_tuple[0]],
                                                              emotion_value=emotion_tuple[1] * 100))

        print(displayed_emotions)

