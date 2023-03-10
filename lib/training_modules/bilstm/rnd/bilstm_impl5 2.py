import numpy as np
from keras import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from numpy import array

from lib.training_modules.bilstm.bilstm import BiLstm


class BiLstmImpl5(BiLstm):
    # def compare_lstm_bi_lstm(self, n_time_steps=10):
    #     print(self.results)

    def run_bi_lstm(self, mode='concat', n_unique_words=10000, max_len=200, batch_size=128, epoch=12):
        data, label, word_index, tokenize = self.get_dataset_value(num_words=n_unique_words, max_len=max_len)
        vocab_size = len(word_index) + 1

        model = self.make_bi_lstm_model(mode, input_emb_dim=vocab_size, max_len=max_len)

        loss, accuracy = self.train_bi_lstm_model(model=model, y_train=label, x_test=data, y_test=label,
                                                  x_train=data,
                                                  batch_size=batch_size, epoch=1)
        print('loss')
        print('loss')

        seq = ['Amazingly bad', 'Wonderful', 'Could be better', 'Did not live up to the excitement']
        train = tokenize.texts_to_sequences(seq)
        d = pad_sequences(train, padding="post", maxlen=max_len)
        self.prediction(model=model, sample=d)

    @staticmethod
    def prediction(model, sample):

        predict = model.predict(sample)
        for p in predict:
            if p == 1:
                print("Positive")
            else:
                print("Negative")

    # def do_lstm(self, n_time_steps, backwards, result_name):
    #     result_name = 'lstm_' + str(result_name)

    # def do_bi_lstm(self, n_time_steps, result_name, mode='concat'):
    #     x_train, y_train, x_test, y_test = self.get_dataset_value(num_words=0, max_len=0)

    # def get_lstm_model(self, n_time_steps, backwards, activation='sigmoid', loss='binary_crossentropy',
    #                    optimizer='adam'):
    #     print('LSTM MODEL ==> n_time_steps is: ' + str(n_time_steps) + '\tbackwards: ' + str(backwards))

    @staticmethod
    def make_bi_lstm_model(mode, input_emb_dim, max_len, output_emb_dim=32, output_dim=128, activation='sigmoid',
                           loss='binary_crossentropy',
                           dropout_rate=0.5,
                           optimizer='adam'):

        model = Sequential()
        model.add(Embedding(input_emb_dim, output_emb_dim))
        model.add(Bidirectional(LSTM(output_dim, return_sequences=True)))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(1, activation=activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        return model

    @staticmethod
    def train_bi_lstm_model(model, x_train, y_train,
                            batch_size,
                            x_test, y_test, epoch=15):
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            # validation_data=[x_test, y_test],
                            )
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        return loss, accuracy

    @staticmethod
    def get_dataset_value(num_words, max_len):
        item = 'baby you are here. i feel are that  	0'
        raws = array([item for _ in range(num_words)])

        data = []
        label = []
        for d in raws:
            arr = d.split("  	")
            data.append(arr[0])
            label.append(arr[1])

        tokenize = Tokenizer(oov_token="<OOV>")
        tokenize.fit_on_texts(data)
        word_index = tokenize.word_index

        train = tokenize.texts_to_sequences(data)
        data = pad_sequences(train, padding="post", maxlen=max_len)

        maxlen = data.shape

        label = np.array(label, dtype='float64')

        return data, label, word_index, tokenize


b = BiLstmImpl5()
b.run_bi_lstm()
