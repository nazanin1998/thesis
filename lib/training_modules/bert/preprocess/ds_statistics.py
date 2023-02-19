from sklearn.model_selection import train_test_split
import tensorflow as tf

from lib.training_modules.bert.bert_configurations import bert_test_size, bert_train_size, bert_val_size
from lib.utils.log.logger import log_phase_desc


class DsStatistics:

    def __init__(self):
        self.__categorical_feature_names = []
        self.__str_feature_names = ['text', "reaction_text"]
        self.__binary_feature_names = ['is_truncated', 'is_source_tweet', 'user.verified', 'user.protected', ]
        self.__numeric_feature_names = ['tweet_id', 'tweet_length', 'symbol_count', 'mentions_count', 'urls_count',
                                        'retweet_count', 'favorite_count', 'hashtags_count', 'in_reply_user_id',
                                        'in_reply_tweet_id', 'user.id', 'user.name_length', 'user.listed_count',
                                        'user.tweets_count', 'user.statuses_count', 'user.friends_count',
                                        'user.favourites_count', 'user.followers_count', 'user.follow_request_sent', ]

        self.__label_feature_name = 'is_rumour'

    def get_categorical_binary_numeric_string_feature_names(self):
        return self.__label_feature_name, self.__categorical_feature_names, self.__binary_feature_names, \
               self.__numeric_feature_names, self.__str_feature_names

    @staticmethod
    def __get_ds_size(df):
        return df.shape[0]

    @staticmethod
    def get_available_splits():
        return list(['train', 'validation', 'test'])

    def get_train_val_test_tensors(
            self,
            df
    ):
        x = df[self.__str_feature_names]
        y = df[self.__label_feature_name]

        label_classes = len(y.value_counts())

        x_train, x_val, x_test, y_train, y_val, y_test = self.train_val_test_split(
            x=x,
            y=y,
            test_size=bert_test_size,
            train_size=bert_train_size,
            val_size=bert_val_size)

        # x_train = x_train.to_records(index=False)
        # print(f"x_train are {x_train}")
        # print(f"x_train are {list(x_train.to_records(index=False))}")
        # print(f"x_train are {type(list(x_train.to_records(index=False)))}")

        x_train_tensor = self.__convert_to_tensor(x_train, dtype=tf.string)
        # print(f"x_train first shape{x_train_tensor.shape}")
        # new_shape = (x_train_tensor.shape[1], x_train_tensor.shape[0])
        # print(f"new_shape {new_shape}")
        # x_train_tensor = tf.reshape(x_train_tensor, new_shape)
        # print(f"new_shape {x_train_tensor[0]}")

        x_val_tensor = self.__convert_to_tensor(x_val, dtype=tf.string)
        x_test_tensor = self.__convert_to_tensor(x_test, dtype=tf.string)
        y_train_tensor = self.__convert_to_tensor(y_train, dtype=tf.int64)
        y_val_tensor = self.__convert_to_tensor(y_val, dtype=tf.int64)
        y_test_tensor = self.__convert_to_tensor(y_test, dtype=tf.int64)

        self.print_ds_statistics(df, label_classes, x_train_tensor, x_val_tensor, x_test_tensor)

        return label_classes, x_train_tensor, x_val_tensor, x_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

    @staticmethod
    def train_val_test_split(x, y, train_size, val_size, test_size):
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size)

        relative_train_size = train_size / (val_size + train_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                          train_size=relative_train_size,
                                                          test_size=1 - relative_train_size)
        return x_train, x_val, x_test, y_train, y_val, y_test

    @staticmethod
    def __convert_to_tensor(
            feature,
            dtype=None):
        return tf.convert_to_tensor(feature, dtype=dtype)

    def print_ds_statistics(
            self,
            df,
            label_classes,
            x_train_tensor,
            x_val_tensor,
            x_test_tensor,
    ):
        log_phase_desc(f'PHEME DS (SIZE)   : {self.__get_ds_size(df)}')
        log_phase_desc(f'LABEL CLASSES     : {label_classes}')
        log_phase_desc(f'TRAINING FEATURE  : {self.__str_feature_names}')
        log_phase_desc(f'LABEL FEATURE     : {self.__label_feature_name}\n')
        log_phase_desc(f'TRAIN      (SIZE) : {x_train_tensor.shape} ({bert_train_size * 100}%)')
        log_phase_desc(f'VALIDATION (SIZE) : {x_val_tensor.shape} ({bert_val_size * 100}%)')
        log_phase_desc(f'TEST       (SIZE) : {x_test_tensor.shape} ({bert_test_size * 100}%)')
