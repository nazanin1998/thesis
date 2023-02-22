from sklearn.model_selection import train_test_split
import tensorflow as tf

from lib.constants import PHEME_LABEL_COL_NAME
from lib.training_modules.bert.bert_configurations import PREPROCESS_TEST_SIZE, PREPROCESS_TRAIN_SIZE, \
    PREPROCESS_VAL_SIZE, \
    PREPROCESS_IGNORE_EXC_STR, PREPROCESS_ONLY_SOURCE_TWEET, PREPROCESS_DO_SHUFFLING
from lib.utils.log.logger import log_phase_desc, print_indented_key_value


class BertPreprocessDsStatistics:

    def __init__(self):
        self.__categorical_feature_names = ['event']
        if PREPROCESS_ONLY_SOURCE_TWEET:
            self.__str_feature_names = ['text']
        else:
            self.__str_feature_names = ['text', "reaction_text"]

        self.__binary_feature_names = ['is_truncated', 'is_source_tweet', 'user.verified', 'user.protected', ]
        self.__numeric_feature_names = ['tweet_length', 'symbol_count', 'mentions_count', 'urls_count',
                                        'retweet_count', 'favorite_count', 'hashtags_count',
                                        'user.name_length', 'user.listed_count',
                                        'user.tweets_count', 'user.statuses_count', 'user.friends_count',
                                        'user.favourites_count', 'user.followers_count', 'user.follow_request_sent', ]

        self.__training_features_name = self.__str_feature_names + self.__binary_feature_names + self.__categorical_feature_names + self.__numeric_feature_names
        if PREPROCESS_IGNORE_EXC_STR:
            self.__training_features_name = self.__str_feature_names

        self.__label_feature_name = PHEME_LABEL_COL_NAME

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
            train_df, val_df, test_df
    ):
        # x = df[self.__training_features_name]
        #
        # y = df[self.__label_feature_name]
        x_train_tensor = train_df[self.__training_features_name]
        x_val_tensor = val_df[self.__training_features_name]
        x_test_tensor = test_df[self.__training_features_name]

        y_train_tensor = train_df[self.__label_feature_name]
        y_val_tensor = val_df[self.__label_feature_name]
        y_test_tensor = test_df[self.__label_feature_name]

        label_classes = len(y_train_tensor.value_counts())

        # x_train_tensor = self.__convert_to_tensor(x_train, dtype=tf.string)
        # x_val_tensor = self.__convert_to_tensor(x_val, dtype=tf.string)
        # x_test_tensor = self.__convert_to_tensor(x_test, dtype=tf.string)
        # y_train_tensor = self.__convert_to_tensor(y_train, dtype=tf.int64)
        # y_val_tensor = self.__convert_to_tensor(y_val, dtype=tf.int64)
        # y_test_tensor = self.__convert_to_tensor(y_test, dtype=tf.int64)

        self.print_ds_statistics(y_train_tensor, y_val_tensor, y_test_tensor, label_classes, x_train_tensor,
                                 x_val_tensor, x_test_tensor)

        return label_classes, x_train_tensor, x_val_tensor, x_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

    # @staticmethod
    # def train_val_test_split(x, y, train_size, val_size, test_size):
    #     x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size,
    #                                                                 shuffle=PREPROCESS_DO_SHUFFLING,
    #                                                                 stratify=None)
    #
    #     relative_train_size = train_size / (val_size + train_size)
    #     x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
    #                                                       train_size=relative_train_size,
    #                                                       test_size=1 - relative_train_size,
    #                                                       shuffle=PREPROCESS_DO_SHUFFLING,
    #                                                       stratify=None)
    #     return x_train, x_val, x_test, y_train, y_val, y_test

    @staticmethod
    def __convert_to_tensor(
            feature,
            dtype=None):
        return tf.convert_to_tensor(feature, dtype=dtype)

    def print_ds_statistics(
            self,
            y_train_tensor,
            y_val_tensor,
            y_test_tensor,
            label_classes,
            x_train_tensor,
            x_val_tensor,
            x_test_tensor
    ):
        log_phase_desc(f'LABEL CLASSES     : {label_classes}')
        log_phase_desc(f'TRAINING FEATURE  : {self.__training_features_name}')
        log_phase_desc(f'LABEL FEATURE     : {self.__label_feature_name}\n')
        log_phase_desc(f'TRAIN      (SIZE) : {x_train_tensor.shape} ({PREPROCESS_TRAIN_SIZE * 100}%)')
        log_phase_desc(f'VALIDATION (SIZE) : {x_val_tensor.shape} ({PREPROCESS_VAL_SIZE * 100}%)')
        log_phase_desc(f'TEST       (SIZE) : {x_test_tensor.shape} ({PREPROCESS_TEST_SIZE * 100}%)')
        print_indented_key_value(f'\tYTrain classes\t  : ', f'{y_train_tensor.value_counts()}', intend_num=6)
        print_indented_key_value(f'\tYVal classes\t  : ', f'{y_val_tensor.value_counts()}', intend_num=6)
        print_indented_key_value(f'\tYTest classes\t  : ', f'{y_test_tensor.value_counts()}', intend_num=6)
