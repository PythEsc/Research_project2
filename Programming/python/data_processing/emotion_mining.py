import os
import traceback

from pycorenlp.corenlp import StanfordCoreNLP

from importer.database.data_types import Post, Comment, Emotion
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

    def tag_posts(self):
        """
        Iterates over all entries in the given DataStorage that do not have a NER tag yet and creates those
        :return: -
        """
        filter = {Post.COLL_EMOTION: {'$exists': False}}
        for post in self.data_storage.iterate_single_post(filter):
            try:
                comment_start_id = post.id.split("_")[1]
                regex_id = comment_start_id + '_.+'
                filter_comment = {Comment.COLL_ID: {'$regex': regex_id}}

                comment_collection = ''
                for comment in self.data_storage.iterate_single_comment(filter_comment, False):
                    comment_collection = comment_collection + os.linesep + comment.content

                # Here we need to mine the emotion of all words in the comment.
                # So we need to lookup the words in the dictionary.
                post.emotion = self.get_post_emotion_value(comment_collection)
                self.data_storage.update_post(post)
            except Exception:
                print("Error while processing post with id: " + post.id)
                traceback.print_exc()


if __name__ == '__main__':
    db = MongodbStorage()
    tagger = EmotionMiner(data_storage=db)
    tagger.tag_posts()
