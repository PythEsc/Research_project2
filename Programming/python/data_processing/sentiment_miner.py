import os
import traceback

from pycorenlp.corenlp import StanfordCoreNLP

from importer.database.data_types import Post, Comment
from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage


class Sentimenter:
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
        self.properties_sentiment = {'annotators': 'sentiment', 'outputFormat': 'json'}

    def get_sentiment_value(self, sentiment_counter):
        if sentiment_counter < -0.5:
            # negative
            result = 'negative'
        elif sentiment_counter > 0.5:
            # positive
            result = 'positive'
        else:
            result = 'neutral'
        return result

    def tag_comments_sentiment_to_posts(self):
        """
        Iterates over all entries in the given DataStorage that do not have a NER tag yet and creates those
        :return: -
        """
        filter = {Post.COLL_COMMENT_SENTIMENT: {'$exists': False}}
        for post in self.data_storage.iterate_single_post(filter):
            try:
                filter_comment = {Comment.COLL_PARENT_ID: post.post_id}
                comment_collection = ''

                for comment in self.data_storage.iterate_single_comment(filter_comment, False):
                    comment_collection = comment_collection + os.linesep + comment.content

                output = self.nlp.annotate(comment_collection, properties=self.properties_sentiment)
                sentences = output['sentences']

                sentiment_counter = 0
                for sentence in sentences:
                    sentiment = sentence['sentiment']
                    if sentiment == 'Negative':
                        sentiment_counter -= 1
                    if sentiment == 'Verynegative':
                        sentiment_counter -= 2
                    elif sentiment == 'Positive':
                        sentiment_counter += 1
                    elif sentiment == 'Verypositive':
                        sentiment_counter += 2

                sentiment_ratio = sentiment_counter / len(sentences)
                post.comment_sentiment = (self.get_sentiment_value(sentiment_ratio), sentiment_ratio)
                self.data_storage.update_post(post)
            except Exception:
                print("Error while processing post with id: " + post.post_id)
                traceback.print_exc()

    def tag_posts_sentiment(self):
        """
        Iterates over all entries in the given DataStorage that do not have a NER tag yet and creates those
        :return: -
        """
        filter = {Post.COLL_SENTIMENT: {'$exists': False}}
        for post in self.data_storage.iterate_single_post(filter):
            try:
                output = self.nlp.annotate(post.message, properties=self.properties_sentiment)
                sentences = output['sentences']

                sentiment_counter = 0
                for sentence in sentences:
                    sentiment = sentence['sentiment']
                    if sentiment == 'Negative':
                        sentiment_counter -= 1
                    if sentiment == 'Verynegative':
                        sentiment_counter -= 2
                    elif sentiment == 'Positive':
                        sentiment_counter += 1
                    elif sentiment == 'Verypositive':
                        sentiment_counter += 2

                sentiment_ratio = sentiment_counter / len(sentences)
                post.sentiment = (self.get_sentiment_value(sentiment_ratio), sentiment_ratio)
                self.data_storage.update_post(post)
            except Exception:
                print("Error while processing post with id: " + post.post_id)
                traceback.print_exc()


if __name__ == '__main__':
    db = MongodbStorage()
    tagger = Sentimenter(data_storage=db)
    tagger.tag_posts_sentiment()
    tagger.tag_comments_sentiment_to_posts()
