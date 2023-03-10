from keras import layers
from tensorflow import keras

# from tensorflow import layers


model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))

from matplotlib import pyplot
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM, Embedding
from keras.layers import Bidirectional
from lib.training_modules.bilstm.bilstm import BiLstm


class BiLstmImpl3(BiLstm):
    def __init__(self):
        self.lstm_model = None
        self.bi_lstm_model = None
        self.results = DataFrame()
        self.max_features = 20000  # Only consider the top 20k words
        self.max_len = 200  # Only consider the first 200 words of each movie review
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    def compare_lstm_bi_lstm(self, n_time_steps=10):
        # self.lstm_model, self.results = self.do_lstm(n_time_steps=n_time_steps, backwards=False, result_name='forward')
        # self.lstm_model, self.results = self.do_lstm(n_time_steps=n_time_steps, backwards=True, result_name='backward')
        self.bi_lstm_model, self.results = self.do_bi_lstm(n_time_steps=n_time_steps, result_name='backward')

        print(self.results)
        self.results.plot()
        pyplot.show()

    def do_lstm(self, n_time_steps, backwards, result_name):
        result_name = 'lstm_' + str(result_name)
        # self.lstm_model = self.get_lstm_model(n_time_steps=n_time_steps, backwards=backwards)
        # self.results[result_name] = self.train_model(model=self.lstm_model, n_time_steps=n_time_steps)
        # return self.lstm_model, self.results

    def do_bi_lstm(self, n_time_steps, result_name, mode='concat'):
        result_name = 'bi_lstm_' + str(result_name)
        self.bi_lstm_model = self.get_bi_lstm_model(n_time_steps=n_time_steps, mode=mode)
        self.results[result_name] = self.train_model(model=self.bi_lstm_model, n_time_steps=n_time_steps)
        return self.bi_lstm_model, self.results

    def get_lstm_model(self, n_time_steps, backwards, activation='sigmoid', loss='binary_crossentropy',
                       optimizer='adam'):
        self.lstm_model = Sequential()
        # self.lstm_model.add(LSTM(20, input_shape=(n_time_steps, 1), return_sequences=True, go_backwards=backwards))
        # self.lstm_model.add(TimeDistributed(Dense(1, activation=activation)))
        # self.lstm_model.compile(loss=loss, optimizer=optimizer)

        # print('LSTM MODEL ==> n_time_steps is: ' + str(n_time_steps) + '\tbackwards: ' + str(backwards))
        # return self.lstm_model

    r"""
        Input for variable-length sequences of integers
        Embed each integer in a 128-dimensional vector
        Add 2 bidirectional LSTMs
        Add a classifier
    """

    def get_bi_lstm_model(self, n_time_steps, mode, activation='sigmoid', loss='binary_crossentropy',
                          optimizer='adam'):
        inputs = keras.Input(shape=(None,), dtype="int32")

        x = Embedding(self.max_features, 128)(inputs)

        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Bidirectional(LSTM(64))(x)

        outputs = layers.Dense(1, activation=activation)(x)
        self.bi_lstm_model = keras.Model(inputs, outputs)
        self.bi_lstm_model.summary()
        print('BiLSTM MODEL ==> n_time_steps is: ' + str(n_time_steps) + '\tmode: ' + str(mode))
        return self.bi_lstm_model

    def train_model(self, model, n_time_steps, epoch=2):
        self.get_train_value()
        model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
        return model

    def get_train_value(self):
        (self.x_train, self.y_train), (self.x_val, self.y_val) = keras.datasets.imdb.load_data(
            num_words=self.max_features
        )
        print(len(self.x_train), "Training sequences")
        print(len(self.x_val), "Validation sequences")
        self.x_train = keras.preprocessing.sequence.pad_sequences(self.x_train, maxlen=self.max_len)
        self.x_val = keras.preprocessing.sequence.pad_sequences(self.x_val, maxlen=self.max_len)
        return self.x_train, self.x_val

    @staticmethod
    def generate_random_seq(n_time_steps):
        pass
        #
        # return X, y
