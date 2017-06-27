from neural_networks.neural_network import NeuralNetwork
from importer.database.database_access import DataStorage
from neural_networks.data_helpers import get_training_set, clean_text
from neural_networks.recurrent_neural_network.text_rnn import TextRNN
from neural_networks.convolutional_neural_network.text_cnn import TextCNN
from pre_trained_embeddings.word_embeddings import WordEmbeddings
from importer.database.mongodb import MongodbStorage
from data_processing.emotion_mining.negation_handling import NegationHandler


from sklearn import metrics
from sklearn.linear_model import LinearRegression
import numpy as np
from tensorflow.contrib import learn

class Regressor():
    def __init__(self, model='LinearRegression'):
        self.model = model

        if self.model == 'LinearRegression':
            self.regressor = LinearRegression()
        else:
            raise ValueError('Unknown Model!')

    def fit(self, db: DataStorage, sample_percentage: float = 0.1):
        print('Started training...')

        # Load data
        print('\nLoading training data...')
        x_text, y = get_training_set(db)
        x_text = clean_text(x_text)

        # Randomly Shuffle the data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = [x_text[i] for i in shuffle_indices]
        y_shuffled = np.copy(y)
        for i in range(len(shuffle_indices)):
            value = y[shuffle_indices[i]]
            y_shuffled[i] = value

        # Split train/dev set
        dev_sample_index = -1 * int(sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("\nTrain/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        print('NN prediction')
        rnn_out = TextCNN.predict(x_train)
        rnn_out_dev = TextCNN.predict(x_dev)

        # print('Emotion miner')
        # nh = NegationHandler(db)
        # nh_out = []
        # for x in x_train:
        #     nh_out.append(nh.get_emotion(x))
        #
        # nh_out_dev = []
        # for i, x in enumerate(x_dev):
        #     nh_out_dev.append(nh.get_emotion(x))
        #     print('\r{:.2f}%'.format(i/len(x_dev)))
        #
        # output = [rnn_out[i] + nh_out[i] for i in range(len(x_train))]
        # output_dev = [rnn_out_dev[i] + nh_out_dev[i] for i in range(len(x_dev))]

        print('regression')
        # self.regressor.fit(output, y_train)
        self.regressor.fit(rnn_out, y_train)

        y_pred = self.regressor.predict(rnn_out_dev)
        # mse = metrics.mean_squared_error(y_dev, output_dev)
        mse = metrics.mean_squared_error(y_dev, y_pred)
        print('MSE: {}'.format(mse))

    def predict(self, content: list) -> list:
        rnn_out = TextRNN.predict(content)

        nh = NegationHandler(db)
        nh_out = []
        for x in content:
            nh_out.append(nh.get_emotion(x))

        output = [rnn_out[i] + nh_out[i] for i in range(len(content))]

        return self.regressor.predict(output)



if __name__ == '__main__':
    db = MongodbStorage()
    reg = Regressor()
    reg.fit(db)

    content = [
        'This is me.',
        'I hate your supermarket!',
        'Your employees are so rude!',
    ]

    print(reg.predict(content))

