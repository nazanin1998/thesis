import numpy as np
from keras import Sequential
from keras.datasets import imdb
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split

from lib.preprocessing.pheme.preprocess_impl import PreProcessImpl
from lib.training_modules.bilstm.bilstm import BiLstm


class BiLstmImpl(BiLstm):

    def __init__(self):
        print('<< PHASE-3 <==> BiLSTM >>')

    def data_reshape(self,
                     df,
                     label_name='is_rumour',
                     col_name='text_pre',
                     test_size=0.2):
        reshaped_df = PreProcessImpl.get_array_column_by_name(df=df,
                                                              col_name=col_name)
        labels = df[label_name]
        x_train, x_test, y_train, y_test = train_test_split(reshaped_df,
                                                            labels,
                                                            test_size=test_size,
                                                            stratify=df[label_name])
        return x_train, x_test, y_train, y_test

    def run_bi_lstm(self,
                    x_train=None,
                    y_train=None,
                    x_test=None,
                    y_test=None,
                    mode='concat',
                    input_emb_dim=10000,
                    max_len=200,
                    output_emb_dim=64,
                    epoch=12,
                    output_dim=64,
                    activation='sigmoid',
                    batch_size=128,
                    optimizer='adam',
                    loss='binary_crossentropy',
                    dropout_rate=0.5):
        if x_train is None:
            x_train, y_train, x_test, y_test = self.get_dataset_value(num_words=input_emb_dim, max_len=max_len)

        model = self.get_bi_lstm_model(mode=mode,
                                       loss=loss,
                                       input_emb_dim=input_emb_dim,
                                       max_len=max_len,
                                       output_dim=output_dim,
                                       activation=activation,
                                       dropout_rate=dropout_rate,
                                       optimizer=optimizer,
                                       output_emb_dim=output_emb_dim)

        loss, accuracy = self.train_bi_lstm_model(model=model,
                                                  y_train=y_train,
                                                  x_test=x_test,
                                                  y_test=y_test,
                                                  x_train=x_train,
                                                  batch_size=batch_size,
                                                  epoch=epoch)
        print('loss is: ' + str(loss))
        print('accuracy is: ' + str(accuracy))
        print('<< PHASE-3 <==> BiLSTM DONE >>')

    def get_bi_lstm_model(self,
                          mode,
                          input_emb_dim,
                          max_len,
                          output_emb_dim=128,
                          output_dim=64,
                          activation='sigmoid',
                          loss='binary_crossentropy',
                          dropout_rate=0.5,
                          optimizer='adam'):
        model = Sequential()
        model.add(Embedding(input_dim=input_emb_dim, output_dim=output_emb_dim, input_length=max_len))
        model.add(Bidirectional(LSTM(output_dim), merge_mode=mode))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation=activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        model.summary()

        return model

    def train_bi_lstm_model(self, model,
                            x_train,
                            y_train,
                            batch_size,
                            x_test,
                            y_test,
                            epoch=1):
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            validation_data=[x_test, y_test])
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        return loss, accuracy

    @staticmethod
    def get_dataset_value(num_words, max_len):
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_len)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        return x_train, y_train, x_test, y_test

    def compare_lstm_bi_lstm(self, n_time_steps=10):
        raise 'Unimplemented method'


def do_bi_lstm(dataframe):
    bi_lstm = BiLstmImpl()
    x_train, x_test, y_train, y_test = bi_lstm.data_reshape(df=dataframe)
    bi_lstm.run_bi_lstm(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, max_len=64, epoch=1)
