from abc import ABC


class BiLstm(ABC):
    def data_reshape(self, df, label_name='is_rumour'):
        pass

    def run_bi_lstm(self, x_train=None, y_train=None, x_test=None, y_test=None, mode='concat', input_emb_dim=10000,
                    max_len=200, output_emb_dim=64, output_dim=64, activation='sigmoid',
                    batch_size=128, epoch=12, optimizer='adam', loss='binary_crossentropy', dropout_rate=0.5):
        pass

    def get_lstm_model(self, n_time_steps, backwards, activation='sigmoid', loss='binary_crossentropy',
                       optimizer='adam'):
        pass

    def train_bi_lstm_model(self,
                            model,
                            x_train,
                            y_train,
                            batch_size,
                            x_test,
                            y_test,
                            epoch=1):
        pass

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
        r"""
        # Arguments
            merge_mode: Mode by which outputs of the
                    forward and backward RNNs will be combined.
                    One of {'sum', 'mul', 'concat', 'ave', None}.
                    If None, the outputs will not be combined,
                    they will be returned as a list.
        """
        pass

    def train_model(self, model,
                    x_train,
                    y_train,
                    batch_size,
                    x_test,
                    y_test,
                    epoch=1):
        pass

    @staticmethod
    def generate_random_seq(n_time_steps):
        r"""
        Generate random sequence:
            create a sequence of random numbers in [0,1]
            limit = calculate cut-off value to change class values
            determine the class outcome for each item in cumulative sequence
            reshape input and output data to be suitable for LSTMs
        """

    def compare_lstm_bi_lstm(self, n_time_steps=10):
        pass
