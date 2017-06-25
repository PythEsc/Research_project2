import os
import pickle
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from importer.database.data_types import Emotion
from importer.database.database_access import DataStorage
from importer.database.mongodb import MongodbStorage


class CategoryClassifier:
    def __init__(self, db: DataStorage):
        self.clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', OneVsRestClassifier(svm.LinearSVC()))])

        self.db = db

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

    def evaluate(self, percent_training: float = 0.95):
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


if __name__ == '__main__':
    db = MongodbStorage()
    cat_clf = CategoryClassifier(db)

    # filename = "../../../../data/category_classifier.pickle"
    #
    # if os.path.isfile(filename):
    #     cat_clf.load_model_from_file(filename)
    # else:
    #     cat_clf.train()
    #     cat_clf.save_model_to_file(filename)

    cat_clf.evaluate(0.95)
