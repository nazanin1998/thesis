from random import random
from numpy import array
from numpy import cumsum
from matplotlib import pyplot
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from lib.training_modules.bilstm.bilstm import BiLstm


class BiLstmImpl2(BiLstm):
    def __init__(self):
        self.lstm_model = None
        self.bi_lstm_model = None
        self.results = DataFrame()

    def compare_lstm_bi_lstm(self, n_time_steps=10):
        self.lstm_model, self.results = self.do_lstm(n_time_steps=n_time_steps, backwards=False, result_name='forward')
        self.lstm_model, self.results = self.do_lstm(n_time_steps=n_time_steps, backwards=True, result_name='backward')
        self.bi_lstm_model, self.results = self.do_bi_lstm(n_time_steps=n_time_steps, result_name='backward')

        print(self.results)
        self.results.plot()
        pyplot.show()

    def do_lstm(self, n_time_steps, backwards, result_name):
        result_name = 'lstm_' + str(result_name)
        self.lstm_model = self.get_lstm_model(n_time_steps=n_time_steps, backwards=backwards)
        self.results[result_name] = self.train_model(model=self.lstm_model, n_time_steps=n_time_steps)
        return self.lstm_model, self.results

    def do_bi_lstm(self, n_time_steps, result_name, mode='concat'):
        result_name = 'bi_lstm_' + str(result_name)
        self.bi_lstm_model = self.get_bi_lstm_model(n_time_steps=n_time_steps, mode=mode)
        self.results[result_name] = self.train_model(model=self.bi_lstm_model, n_time_steps=n_time_steps)
        return self.bi_lstm_model, self.results

    def get_lstm_model(self, n_time_steps, backwards, activation='sigmoid', loss='binary_crossentropy',
                       optimizer='adam'):
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(20, input_shape=(n_time_steps, 1), return_sequences=True, go_backwards=backwards))
        self.lstm_model.add(TimeDistributed(Dense(1, activation=activation)))
        self.lstm_model.compile(loss=loss, optimizer=optimizer)

        print('LSTM MODEL ==> n_time_steps is: ' + str(n_time_steps) + '\tbackwards: ' + str(backwards))
        return self.lstm_model

    r"""
        # Arguments
            merge_mode: Mode by which outputs of the
                forward and backward RNNs will be combined.
                One of {'sum', 'mul', 'concat', 'ave', None}.
                If None, the outputs will not be combined,
                they will be returned as a list.
    """

    def get_bi_lstm_model(self, n_time_steps, mode, activation='sigmoid', loss='binary_crossentropy',
                          optimizer='adam'):
        self.bi_lstm_model = Sequential()
        self.bi_lstm_model.add(
            Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_time_steps, 1), merge_mode=mode))
        self.bi_lstm_model.add(TimeDistributed(Dense(1, activation=activation)))
        self.bi_lstm_model.compile(loss=loss, optimizer=optimizer)
        print('BiLSTM MODEL ==> n_time_steps is: ' + str(n_time_steps) + '\tmode: ' + str(mode))
        return self.bi_lstm_model

    def train_model(self, model, n_time_steps, epoch=250):
        loss = list()
        for _ in range(epoch):
            X, y = self.generate_random_seq(n_time_steps=n_time_steps)
            hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
            loss.append(hist.history['loss'][0])
        return loss

    r"""
        Generate random sequence:
            create a sequence of random numbers in [0,1]
            limit = calculate cut-off value to change class values
            determine the class outcome for each item in cumulative sequence
            reshape input and output data to be suitable for LSTMs
    """

    @staticmethod
    def generate_random_seq(n_time_steps):
        limit = n_time_steps / 4.0

        X = array([random() for _ in range(n_time_steps)])
        y = array([0 if x < limit else 1 for x in cumsum(X)])

        X = X.reshape(1, n_time_steps, 1)
        y = y.reshape(1, n_time_steps, 1)

        return X, y
