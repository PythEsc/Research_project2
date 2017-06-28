import os
import pickle
import traceback
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from pycorenlp import StanfordCoreNLP
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from importer.database.data_types import Emotion, Sentence, Comment, Post
from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage


class CategoryClassifier:
    def __init__(self, db: DataStorage):
        self.clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', OneVsRestClassifier(svm.LinearSVC()))])

        self.db = db
        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.properties_ssplit = {'annotators': 'ssplit', 'outputFormat': 'json'}
        self.properties_pos = {'annotators': 'pos', 'outputFormat': 'json'}

    def load_model_from_file(self, filename: str):
        with open(filename, "rb") as input_file:
            self.clf = pickle.load(input_file)

    def save_model_to_file(self, filename: str):
        with open(filename, "wb") as output_file:
            pickle.dump(self.clf, output_file)

    def train(self):
        data = []
        labels = []
        # for sentence in self.db.iterate_single_sentence({}):
        #     data.append(sentence.content)
        #
        #     emotions = []
        #     for index, emotion in enumerate(sentence.emotion):
        #         if emotion > 0:
        #             emotions.append(Emotion.EMOTION_TYPES[index])
        #
        #     labels.append(emotions)

        for sentence in self.db.iterate_single_sentence({}):
            data.append(sentence.content)
            labels.append(np.array([int(x > 0) for x in sentence.emotion]))

        # mlb = MultiLabelBinarizer()
        # y_enc = mlb.fit_transform(labels)

        self.clf.fit(data, np.array(labels))

    def classify(self, content: str) -> list:
        return self.clf.predict([content])

    def evaluate(self, percent_training: float = 0.95, use_pickle: bool = False):
        data = []
        labels = []
        for sentence in self.db.iterate_single_sentence({}):
            data.append(sentence.content)
            labels.append(np.array([int(x > 0) for x in sentence.emotion]))

        size = len(data)
        per = int(percent_training * size)
        x_train = data[:per]
        x_test = data[per:]
        y_train = np.array(labels[:per])
        y_test = np.array(labels[per:])

        if use_pickle is False:
            self.clf.fit(x_train, y_train)

        self.plot_accuracy(Emotion.EMOTION_TYPES, len(Emotion.EMOTION_TYPES), x_test, y_test)

    def plot_accuracy(self, categories, n_classes, x_test, y_test):
        count_right = 0
        count_wrong = 0
        predict_labels = self.clf.predict(x_test)
        for i, predict_label in enumerate(predict_labels):
            if (predict_label == y_test[i]).all():
                count_right += 1
            else:
                count_wrong += 1
        print("\n{per: .2f}%".format(per=(count_right / (count_wrong + count_right) * 100)))
        # output the graph for precision recall (using the method of sklearn for that)
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                                predict_labels[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], predict_labels[:, i])

        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                        predict_labels.ravel())
        average_precision["micro"] = average_precision_score(y_test, predict_labels)
        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        lw = 2
        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall[0], precision[0], lw=lw, color='navy',
                 label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
        plt.legend(loc="lower left")
        plt.show()
        # Plot Precision-Recall curve for each class
        plt.clf()
        plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
                 label='average Precision-recall curve (area = {0:0.2f})'
                       ''.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=lw,
                     label='Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(categories[i], average_precision[i]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    """
        This method calculates the emotions for a post AND also uses the SVM to predict the emotions for a 
        non-annotated sentence. 
    """
    def calculate_emotion_for_post(self):
        count_non_annotated = 0
        emotions_statistics = [0] * len(Emotion.EMOTION_TYPES)
        #'Amazon', 'bestbuy'
        filter = {Post.COLL_COMMENT_EMOTION: {'$exists': False}}
        for post in self.db.iterate_single_post(filter):
            try:
                filter_comment = {Comment.COLL_PARENT_ID: post.post_id}
                emotions_post = [0] * len(Emotion.EMOTION_TYPES)

                for comment_object in self.db.iterate_single_comment(filter_comment, False):
                    comment = comment_object.content
                    output = self.nlp.annotate(comment, self.properties_pos)

                    if not isinstance(output, dict):
                        continue
                    sentences = []

                    for sentence in output.get("sentences", []):
                        begin = sentence["tokens"][0]
                        end = sentence["tokens"][-1]

                        start_index = begin['characterOffsetBegin']
                        end_index = end['characterOffsetEnd']
                        sentence_splitted = comment[start_index: end_index]
                        sentences.append(sentence_splitted)
                    # Because I am running already through everything, I can already calculate the overall emotion distribution
                    for sent_index, sentence in enumerate(output.get("sentences", [])):
                        actual_sentence = sentences[sent_index]
                        lookedup_sentence = db.select_single_sentence({Sentence.COLL_CONTENT: actual_sentence})
                        if lookedup_sentence is not None:
                            emotions_post = [x + y for x, y in zip(emotions_post, lookedup_sentence.emotion)]
                            continue
                        sentence_emotion = self.clf.predict([actual_sentence])
                        if len(sentence_emotion) > 0:
                            emotions_statistics = [x + y for x, y in zip(emotions_statistics, sentence_emotion)]
                            emotions_post = [x + y for x, y in zip(emotions_post, sentence_emotion)]
                            count_non_annotated += 1
                            new_sentence = Sentence.create_from_single_values(actual_sentence, sentence_emotion[0].tolist(), True)
                            db.insert_sentence(new_sentence)
                if len(emotions_post) == 1:
                    emotions_post = emotions_post[0].tolist()
                post.comment_emotion = emotions_post
                self.db.update_post(post)
            except Exception:
                print("Error while processing post with id: " + post.post_id)
                traceback.print_exc()
        return count_non_annotated, emotions_statistics

if __name__ == '__main__':
    db = MongodbStorage()

    cc = CategoryClassifier(db)
    cc.load_model_from_file("category_classifier.pickle")
    cna, ems = cc.calculate_emotion_for_post()
    print(cna)
    print(ems)
    # cat_clf = CategoryClassifier(db)
    #
    # filename = "category_classifier.pickle"
    #
    # if os.path.isfile(filename):
    #  cat_clf.load_model_from_file(filename)
    # else:
    #  cat_clf.train()
    #  cat_clf.save_model_to_file(filename)
    #
    # count_non_annotated, emotions_statistics = cat_clf.calculate_emotion_for_post()
    # print("Number of predicted non-annotated sentences: ", str(count_non_annotated))
    # sum_of_emotions = sum(emotions_statistics)
    # emotions_statistics_normalized = [x/sum_of_emotions for x in emotions_statistics]
    # print("Emotion distribution of predicted non-annotated sentences: ", str(emotions_statistics))
    # print("Emotion distribution of predicted non-annotated sentences (normalized): ", str(emotions_statistics_normalized))
    # #cat_clf.evaluate(0.95, use_pickle=False)

    # Sound when finished
    import platform

    if platform.system() == 'Windows':
        import winsound

        for _ in range(5):
            winsound.Beep(500, 200)
