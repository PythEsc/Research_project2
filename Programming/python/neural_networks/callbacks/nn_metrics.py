import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class NNMetric(Callback):
    def __init__(self):
        super().__init__()
        self.precision = []
        self.recall = []
        self.f1 = []

        self.cur_precision = []
        self.cur_recall = []
        self.cur_f1 = []

    def on_train_begin(self, logs=None):
        self.cur_precision = []
        self.cur_recall = []
        self.cur_f1 = []

    def on_epoch_end(self, epoch, logs=None):
        num_classes = len(self.validation_data[1][0])
        threshold = 1 / num_classes

        val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        val_targ = np.asarray(self.validation_data[1])

        val_predict_classes = np.empty(shape=(len(val_predict), num_classes), dtype=bool)
        for index, array in enumerate(val_predict):
            val_predict_classes[index] = np.array([True if value > threshold else False for value in array], dtype=bool)

        val_targ_classes = np.empty(shape=(len(val_targ), num_classes), dtype=bool)
        for index, array in enumerate(val_targ):
            val_targ_classes[index] = np.array([True if value > threshold else False for value in array], dtype=bool)

        precision = precision_score(y_true=val_targ_classes, y_pred=val_predict_classes, average='micro')
        recall = recall_score(y_true=val_targ_classes, y_pred=val_predict_classes, average='micro')
        f1 = f1_score(y_true=val_targ_classes, y_pred=val_predict_classes, average='micro')

        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)
        self.cur_precision.append(precision)
        self.cur_recall.append(recall)
        self.cur_f1.append(f1)

    def on_train_end(self, logs=None):
        print("------------- Training Finished -------------")
        print("Current precision: %.4f , Avg. precision: %.4f (+/- %.4f)" % (
            float(np.mean(self.cur_precision)),
            float(np.mean(self.precision)),
            float(np.std(self.precision))))
        print("Current recall: %.4f , Avg. recall: %.4f (+/- %.4f)" % (float(np.mean(self.cur_recall)),
                                                                                 float(np.mean(self.recall)),
                                                                                 float(np.std(self.recall))))
        print("Current f1-score: %.4f , Avg. f1-score: %.4f (+/- %.4f)" % (float(np.mean(self.cur_f1)),
                                                                                     float(np.mean(self.f1)),
                                                                                     float(np.std(self.f1))))
