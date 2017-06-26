import nltk
from pycorenlp.corenlp import StanfordCoreNLP

from importer.database.data_types import Emotion, Sentence
from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage


class NegationHandler:
    NEGATION_LIST = ["no", "not", "rather", "wont", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere",
                     "cannot", "without", "n't"]
    NEGATION_PREFIX = ["a", "de", "dis", "il", "im", "in", "ir", "mis", "non", "un"]
    NEGATION_SUFFIX = ["less"]
    DIMINISHER = {}
    INTENSIFIER = {"absolutely", "completely", "extremely", "highly", "rather", "really", "so", "too", "totally",
                   "utterly", "very", "at all"}

    EMOTION_OPPOSITES = [[0, 0, 0, 0, 1, 0, 0, 0],  # Anger
                         [0, 0, 0, 1, 0, 0, 1, 0],  # Anticipation
                         [0, 0, 0, 0, 1, 0, 0, 1],  # Disgust
                         [0, 0, 0, 0, 1, 0, 0, 1],  # Fear
                         [1, 0, 1, 1, 0, 1, 0, 0],  # Joy
                         [0, 0, 0, 0, 1, 0, 0, 0],  # Sadness
                         [0, 1, 0, 0, 0, 0, 0, 1],  # Surprise
                         [0, 0, 1, 0, 0, 0, 1, 0]]  # Trust

    def __init__(self, data_storage: DataStorage):
        """
        Const

        :param data_storage: The DataStorage implementation that contains the table with annotated emotions
        """

        self.db = data_storage
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.properties_pos = {'annotators': 'pos', 'outputFormat': 'json'}
        self.properties_ssplit = {'annotators': 'ssplit', 'outputFormat': 'json'}

    def get_emotion(self, content: str):
        total_emotion = [0] * len(Emotion.EMOTION_TYPES)

        output = self.nlp.annotate(content, self.properties_pos)
        if not isinstance(output, dict):
            return
        sentences = []

        for sentence in output.get("sentences", []):
            beginn = sentence["tokens"][0]
            end = sentence["tokens"][-1]

            start_index = beginn['characterOffsetBegin']
            end_index = end['characterOffsetEnd']
            sentence_splitted = content[start_index: end_index]
            sentences.append(sentence_splitted)

        for sent_index, sentence in enumerate(output.get("sentences", [])):
            lookedup_sentence = self.db.select_single_sentence({Sentence.COLL_CONTENT: sentences[sent_index]})
            if lookedup_sentence is not None:
                total_emotion = [x + y for x, y in zip(lookedup_sentence.emotion, total_emotion)]
                continue

            sentence_emotion = [0] * len(Emotion.EMOTION_TYPES)
            add_to_database = False

            index_to_continue = 0
            tokens = sentence["tokens"]
            for token_index, token in enumerate(tokens):
                if token_index < index_to_continue:
                    continue

                if token["originalText"] in self.NEGATION_LIST:
                    if token_index + 1 > len(tokens) - 1:
                        break

                    next_token = tokens[token_index + 1]
                    while True:
                        next_token_emotion = self.db.select_single_emotion(next_token["originalText"])
                        index_to_continue = next_token["index"]
                        if next_token_emotion is not None:
                            negated_emotion = self.__get_negated_emotions(next_token_emotion)
                            sentence_emotion = [x + y for x, y in zip(negated_emotion, sentence_emotion)]
                            add_to_database = True
                            break
                        else:
                            next_token_pos = next_token["pos"]
                            if next_token_pos in ["RB", "VBN"]:
                                if index_to_continue > len(tokens) - 1:
                                    break
                                next_token = tokens[index_to_continue]
                            else:
                                break
                else:
                    token_emotion = self.db.select_single_emotion(token["originalText"])
                    if token_emotion is not None:
                        sentence_emotion = [x + y for x, y in zip(token_emotion.emotion, sentence_emotion)]
                        add_to_database = True

            total_emotion = [x + y for x, y in zip(sentence_emotion, total_emotion)]
            if add_to_database:
                self.db.insert_sentence(Sentence.create_from_single_values(sentences[sent_index], sentence_emotion))

        return total_emotion

    def __get_negated_emotions(self, emotion: Emotion) -> list:
        emotion_word = emotion.id
        emotions = emotion.emotion

        emotion = [0] * len(Emotion.EMOTION_TYPES)
        counter = 0

        for negation_prefix in self.NEGATION_PREFIX:
            word_with_suffix = negation_prefix + emotion_word
            word_with_suffix_emotion = self.db.select_single_emotion({"_id": word_with_suffix})
            if word_with_suffix_emotion is not None:
                emotion = [x + y for x, y in zip(emotion, word_with_suffix_emotion.emotion)]
                counter += 1

        for negation_suffix in self.NEGATION_SUFFIX:
            word_with_suffix = emotion_word + negation_suffix
            word_with_suffix_emotion = self.db.select_single_emotion({"_id": word_with_suffix})
            if word_with_suffix_emotion is not None:
                emotion = [x + y for x, y in zip(emotion, word_with_suffix_emotion.emotion)]
                counter += 1

        if counter != 0:
            emotion = [x / counter for x in emotion]
        else:
            for index, value in enumerate(emotions):
                if value != 0:
                    negated_inner_emotion = [emotion_type * value / sum(self.EMOTION_OPPOSITES[index]) for emotion_type
                                             in
                                             self.EMOTION_OPPOSITES[index]]
                    emotion = [x + y for x, y in zip(emotion, negated_inner_emotion)]
        return emotion


if __name__ == '__main__':
    db = MongodbStorage()
    handler = NegationHandler(db)
    for comment in db.iterate_single_comment({}):
        handler.get_emotion(comment.content)
