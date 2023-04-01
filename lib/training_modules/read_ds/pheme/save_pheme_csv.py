import os

import pandas as pd
from sklearn.model_selection import train_test_split

from lib import constants
from lib.preprocessing.simple_preprocessing.simple_preprocess import SimplePreprocess
from lib.training_modules.basic_reading_ds.basic_reading_ds import my_train_val_test_split
from lib.training_modules.bert.bert_configurations import PREPROCESS_ONLY_SOURCE_TWEET, BERT_TEST_SIZE, BERT_TRAIN_SIZE, \
    BERT_VAL_SIZE, PREPROCESS_DO_SHUFFLING, BERT_K_FOLD


class SavePhemeCsv:
    def __init__(self):
        self.__simple_preprocess = SimplePreprocess()
        self.__train_path = get_train_path_for_specified_split_size()
        self.__val_path = get_val_path_for_specified_split_size()
        self.__test_path = get_test_path_for_specified_split_size()

    def extract_csv_from_events(self, events):
        tweets = self.__extract_tweet_list_from_events(events)

        df = pd.DataFrame(tweets)
        df = self.__simple_preprocess.preprocess(df=df, col_names=[constants.PHEME_TOTAL_TEXT_COL_NAME])

        x = df[:]
        y = x.pop(constants.PHEME_LABEL_COL_NAME)

        x_train, x_val, x_test, y_train, y_val, y_test = my_train_val_test_split(
            x=x,
            y=y,
            test_size=BERT_TEST_SIZE,
            train_size=BERT_TRAIN_SIZE,
            val_size=BERT_VAL_SIZE)

        train_df = x_train[:]
        val_df = x_val[:]
        test_df = x_test[:]

        train_df = train_df.join(y_train)
        val_df = val_df.join(y_val)
        test_df = test_df.join(y_test)

        self.__make_directory_for_specified_split_size()
        train_df.to_csv(self.__train_path, index=False)
        val_df.to_csv(self.__val_path, index=False)
        test_df.to_csv(self.__test_path, index=False)

        return train_df, val_df, test_df

    @staticmethod
    def __make_directory_for_specified_split_size():
        os.makedirs(get_directory_for_specified_split_size(), exist_ok=True)

    @staticmethod
    def __extract_tweet_list_from_events(events):
        tweets = []
        for event in events:

            for rumour in event.rumors:
                if PREPROCESS_ONLY_SOURCE_TWEET and rumour.source_tweet is not None:
                    tweets.append(
                        rumour.source_tweet.to_json(is_rumour=0, event=event.name, is_source_tweet=0, reaction_text='',
                                                    reactions=rumour.reactions))

                if not PREPROCESS_ONLY_SOURCE_TWEET:
                    for reaction in rumour.reactions:
                        tweets.append(
                            reaction.to_json(is_rumour=0, event=event.name, is_source_tweet=1,
                                             reaction_text=reaction.text))

            for non_rumour in event.non_rumors:
                if PREPROCESS_ONLY_SOURCE_TWEET and non_rumour.source_tweet is not None:
                    tweets.append(non_rumour.source_tweet.to_json(is_rumour=1, event=event.name, is_source_tweet=0,
                                                                  reaction_text='', reactions=non_rumour.reactions))
                if not PREPROCESS_ONLY_SOURCE_TWEET:
                    for reaction in non_rumour.reactions:
                        tweets.append(
                            reaction.to_json(is_rumour=1, event=event.name, is_source_tweet=1,
                                             reaction_text=reaction.text))

        return tweets


def get_directory_for_specified_split_size():
    specified_split_dir = f"/{round(BERT_TRAIN_SIZE * 100)}_{round(BERT_VAL_SIZE * 100)}_{round(BERT_TEST_SIZE * 100)}"
    return constants.PHEME_CSV_DIR + specified_split_dir


def get_k_fold_path():
    return constants.PHEME_CSV_DIR + "/" + str(BERT_K_FOLD) + constants.PHEME_K_FOLD_CSV_NAME


def get_train_path_for_specified_split_size():
    return get_directory_for_specified_split_size() + "/" + constants.PHEME_TRAIN_CSV_NAME


def get_val_path_for_specified_split_size():
    return get_directory_for_specified_split_size() + "/" + constants.PHEME_VAL_CSV_NAME


def get_test_path_for_specified_split_size():
    return get_directory_for_specified_split_size() + "/" + constants.PHEME_TEST_CSV_NAME
