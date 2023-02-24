from sklearn.model_selection import train_test_split
from tabulate import tabulate

from lib.read_datasets.pheme.read_pheme_json_ds.read_pheme_json_ds import ReadPhemeJsonDS
import os
import pandas as pd

import lib.constants as constants
from lib.models.event_model import EventModel
from lib.models.tweet_model import TweetModel
from lib.models.tweet_tree_model import TweetTreeModel
from lib.utils.file_dir_handler import FileDirHandler
from lib.training_modules.bert.bert_configurations import PREPROCESS_ONLY_SOURCE_TWEET, BERT_TEST_SIZE, \
    BERT_TRAIN_SIZE, BERT_VAL_SIZE, PREPROCESS_DO_SHUFFLING


class ReadPhemeJsonDSImpl(ReadPhemeJsonDS):
    def __init__(self):
        self.df = None
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.events = None
        self.directory = constants.PHEME_JSON_DIR

    def read_and_save_csv(self):
        print(f"\tRead PHEME dataset (.json) ... directory => {constants.PHEME_JSON_DIR}")
        self.events = self.__extract_events_from_json_dataset()
        self.__extract_csv_from_events()
        self.print_summery()
        return self.train_df, self.val_df, self.test_df

    def print_summery(self):
        index = 0
        l = []

        for event in self.events:
            index += 1
            l.append(event.to_table_array())
        table = tabulate(l, headers=['event title', 'rumours', 'non_rumours', 'rumours', "all_non_rumours"],
                         tablefmt='orgtbl')

        print(table)

    def __extract_events_from_json_dataset(self):
        events = []
        event_dirs = FileDirHandler.read_directories(directory=self.directory)

        for event_dir in event_dirs:
            if event_dir.startswith("."):
                event_dirs.remove(event_dir)

        for event_dir in event_dirs:
            event = self.__extract_event_from_dir(event_dir=self.directory + event_dir)
            if event is not None:
                events.append(event)
        return events

    def __extract_event_from_dir(self, event_dir):
        inner_event_dirs = FileDirHandler.read_directories(directory=event_dir)
        if inner_event_dirs is None:
            return None

        event = EventModel(path=event_dir, rumors=[], non_rumors=[])
        for inner_event_dir in inner_event_dirs:
            tweet_tree_dir = event_dir + '/' + inner_event_dir
            if inner_event_dir == constants.NON_RUMOURS:
                event.non_rumors = self.__tweet_trees_extraction(tweet_tree_dir=tweet_tree_dir)

            elif inner_event_dir == constants.RUMOURS:
                event.rumors = self.__tweet_trees_extraction(tweet_tree_dir=tweet_tree_dir)

        return event

    def __tweet_trees_extraction(self, tweet_tree_dir):
        tweet_trees = []

        tweet_tree_ids = FileDirHandler.read_directories(directory=tweet_tree_dir)

        if tweet_tree_ids is None:
            return None

        for tweet_tree_id in tweet_tree_ids:
            tweet_tree_path = tweet_tree_dir + '/' + tweet_tree_id

            source_tweet_path = tweet_tree_path + '/source-tweets/' + tweet_tree_id + '.json'
            source_tweet_obj = TweetModel.tweet_file_to_obj(path=source_tweet_path)

            reaction_dir = tweet_tree_path + '/reactions/'
            reaction_ids = FileDirHandler.read_directories(reaction_dir)

            reactions = []
            if reaction_ids is not None:
                for reaction_id in reaction_ids:
                    reaction_path = reaction_dir + reaction_id

                    reactions.append(TweetModel.tweet_file_to_obj(path=reaction_path))

            tweet_trees.append(TweetTreeModel(source_tweet=source_tweet_obj, reactions=reactions))

        return tweet_trees

    @staticmethod
    def train_val_test_split(x, y, train_size, val_size, test_size):

        x_train, x_test_val, y_train, y_test_val = train_test_split(x, y, train_size=train_size,
                                                                    shuffle=PREPROCESS_DO_SHUFFLING,
                                                                    stratify=None)

        relative_val_size = val_size / (val_size + test_size)
        x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val,
                                                        train_size=relative_val_size,
                                                        test_size=1 - relative_val_size,
                                                        shuffle=True)
        return x_train, x_val, x_test, y_train, y_val, y_test

    def __extract_csv_from_events(self):
        tweets = self.__extract_tweet_list_from_events()
        self.df = pd.DataFrame(tweets)
        x = self.df[:]
        y = x.pop(constants.PHEME_LABEL_COL_NAME)

        x_train, x_val, x_test, y_train, y_val, y_test = self.train_val_test_split(
            x=x,
            y=y,
            test_size=BERT_TEST_SIZE,
            train_size=BERT_TRAIN_SIZE,
            val_size=BERT_VAL_SIZE)

        self.train_df = x_train[:]
        self.val_df = x_val[:]
        self.test_df = x_test[:]

        self.train_df = self.train_df.join(y_train)
        self.val_df = self.val_df.join(y_val)
        self.test_df = self.test_df.join(y_test)

        self.__make_directory_for_specified_split_size()
        if PREPROCESS_ONLY_SOURCE_TWEET:
            self.df.to_csv(constants.PHEME_CSV_ONLY_TEXT_PATH, index=False)
            # self.train_df.to_csv(constants.PHEME_CSV_ONLY_TEXT_PATH, index=False)
            # self.val_df.to_csv(constants.PHEME_CSV_ONLY_TEXT_PATH, index=False)
            # self.test_df.to_csv(constants.PHEME_CSV_ONLY_TEXT_PATH, index=False)
        else:
            # self.df.to_csv(get_val_path_for_specified_split_size(), index=False)
            self.train_df.to_csv(get_train_path_for_specified_split_size(), index=False)
            self.val_df.to_csv(get_val_path_for_specified_split_size(), index=False)
            self.test_df.to_csv(get_test_path_for_specified_split_size(), index=False)

    @staticmethod
    def __make_directory_for_specified_split_size():
        os.makedirs(get_directory_for_specified_split_size(), exist_ok=True)

    def __extract_tweet_list_from_events(self):
        tweets = []
        for event in self.events:

            for rumour in event.rumors:
                if PREPROCESS_ONLY_SOURCE_TWEET and rumour.source_tweet is not None:
                    tweets.append(
                        rumour.source_tweet.to_json(is_rumour=0, event=event.name, is_source_tweet=0, reaction_text=''))
                if not PREPROCESS_ONLY_SOURCE_TWEET:
                    for reaction in rumour.reactions:
                        tweets.append(
                            reaction.to_json(is_rumour=0, event=event.name, is_source_tweet=1,
                                             reaction_text=reaction.text))

            for non_rumour in event.non_rumors:
                if PREPROCESS_ONLY_SOURCE_TWEET and non_rumour.source_tweet is not None:
                    tweets.append(non_rumour.source_tweet.to_json(is_rumour=1, event=event.name, is_source_tweet=0,
                                                                  reaction_text=''))
                if not PREPROCESS_ONLY_SOURCE_TWEET:
                    for reaction in non_rumour.reactions:
                        tweets.append(
                            reaction.to_json(is_rumour=1, event=event.name, is_source_tweet=1,
                                             reaction_text=reaction.text))

        return tweets


def get_directory_for_specified_split_size():
    specified_split_dir = f"/{round(BERT_TRAIN_SIZE * 100)}_{round(BERT_VAL_SIZE * 100)}_{round(BERT_TEST_SIZE * 100)}"
    return constants.PHEME_CSV_DIR + specified_split_dir


def get_train_path_for_specified_split_size():
    return get_directory_for_specified_split_size() + "/" + constants.PHEME_TRAIN_CSV_NAME


def get_val_path_for_specified_split_size():
    return get_directory_for_specified_split_size() + "/" + constants.PHEME_VAL_CSV_NAME


def get_test_path_for_specified_split_size():
    return get_directory_for_specified_split_size() + "/" + constants.PHEME_TEST_CSV_NAME
