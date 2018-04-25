import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error


class NNMetric(Callback):
    def __init__(self):
        super().__init__()
        self.precision = []
        self.recall = []
        self.f1 = []
        self.mse = []

        self.cur_precision = []
        self.cur_recall = []
        self.cur_f1 = []
        self.cur_mse = []

    def on_train_begin(self, logs=None):
        self.cur_precision = []
        self.cur_recall = []
        self.cur_f1 = []
        self.cur_mse = []

    def on_epoch_end(self, epoch, logs=None):
        num_classes = len(self.validation_data[1][0])

        val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        val_targ = np.asarray(self.validation_data[1])

        precision, recall, f1, mse = NNMetric.evaluate(val_predict, val_targ)

        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)
        self.mse.append(mse)
        self.cur_precision.append(precision)
        self.cur_recall.append(recall)
        self.cur_f1.append(f1)
        self.cur_mse.append(mse)

    @staticmethod
    def evaluate(val_predict, val_targ):
        num_classes = len(val_targ[0])
        threshold = 1 / num_classes

        val_predict_classes = np.empty(shape=(len(val_predict), num_classes), dtype=np.bool)
        for index, array in enumerate(val_predict):
            val_predict_classes[index] = np.array([True if value > threshold else False for value in array], dtype=np.bool)

        val_targ_classes = np.empty(shape=(len(val_targ), num_classes), dtype=np.bool)
        for index, array in enumerate(val_targ):
            val_targ_classes[index] = np.array([True if value > threshold else False for value in array], dtype=np.bool)

        precision = precision_score(y_true=val_targ_classes, y_pred=val_predict_classes, average='micro')
        recall = recall_score(y_true=val_targ_classes, y_pred=val_predict_classes, average='micro')
        f1 = f1_score(y_true=val_targ_classes, y_pred=val_predict_classes, average='micro')

        mse = mean_squared_error(y_true=val_targ, y_pred=val_predict)

        return precision, recall, f1, mse

    def on_train_end(self, logs=None):
        print("------------- Training Finished -------------")
        print("Current MSE: %.4f, Avg. MSE: %.4f (+/- %.4f)" % (
            float(np.mean(self.cur_mse)), float(np.mean(self.mse)), float(np.std(self.mse))))
        print("Current precision: %.4f, Avg. precision: %.4f (+/- %.4f)" % (
            float(np.mean(self.cur_precision)),
            float(np.mean(self.precision)),
            float(np.std(self.precision))))
        print("Current recall: %.4f, Avg. recall: %.4f (+/- %.4f)" % (float(np.mean(self.cur_recall)),
                                                                      float(np.mean(self.recall)),
                                                                      float(np.std(self.recall))))
        print("Current f1-score: %.4f, Avg. f1-score: %.4f (+/- %.4f)" % (float(np.mean(self.cur_f1)),
                                                                          float(np.mean(self.f1)),
                                                                          float(np.std(self.f1))))
