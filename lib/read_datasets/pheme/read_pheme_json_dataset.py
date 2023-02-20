import os
import pandas as pd

import lib.constants as constants
from lib.models.event_model import EventModel
from lib.models.tweet import Tweet
from lib.models.tweet_tree import TweetTree
from lib.read_datasets.pheme.file_dir_handler import FileDirHandler
from lib.training_modules.bert.bert_configurations import only_source_tweet
from lib.utils.log.logger import log_phase_desc
from tabulate import tabulate


class ReadPhemeJsonDataset:
    def __init__(self):
        self.df = None
        self.events = None
        self.directory = constants.PHEME_DIR

    def read_and_save_csv(self):
        print("\tRead dataset (.json) ...")
        print("\tDir (.json): " + constants.PHEME_DIR)
        self.events = self.__extract_events_from_json_dataset()
        self.__extract_csv_from_events()
        self.print_summery()
        return self.df

    def print_summery(self):
        index = 0
        l = []

        for event in self.events:
            index += 1
            l.append(event.to_table_array())
        table = tabulate(l, headers=['event title', 'rumours', 'non_rumours', 'rumours',"all_non_rumours"], tablefmt='orgtbl')

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
            source_tweet_obj = Tweet.tweet_file_to_obj(path=source_tweet_path)

            reaction_dir = tweet_tree_path + '/reactions/'
            reaction_ids = FileDirHandler.read_directories(reaction_dir)

            reactions = []
            if reaction_ids is not None:
                for reaction_id in reaction_ids:
                    reaction_path = reaction_dir + reaction_id

                    reactions.append(Tweet.tweet_file_to_obj(path=reaction_path))

            tweet_trees.append(TweetTree(source_tweet=source_tweet_obj, reactions=reactions))

        return tweet_trees

    def __extract_csv_from_events(self):
        tweets = self.__extract_tweet_list_from_events()
        self.df = pd.DataFrame(tweets)

        os.makedirs(constants.PHEME_CSV_DIR, exist_ok=True)
        if only_source_tweet:
            self.df.to_csv(constants.PHEME_CSV_ONLY_TEXT_PATH, index=False)
        else:
            self.df.to_csv(constants.PHEME_CSV_PATH, index=False)

    def __extract_tweet_list_from_events(self):
        tweets = []
        for event in self.events:

            for rumour in event.rumors:
                # if rumour.source_tweet is not None:
                # else:
                #     print('source tweet is non')

                for reaction in rumour.reactions:
                    tweets.append(rumour.source_tweet.to_json(is_rumour=0, event=event.name, is_source_tweet=1, reaction_text=reaction.text))
                    # tweets.append(reaction.to_json(is_rumour=0, event=event.name, is_source_tweet=1))

            for non_rumour in event.non_rumors:
                # if non_rumour.source_tweet is not None:
                #     tweets.append(non_rumour.source_tweet.to_json(is_rumour=1, event=event.name, is_source_tweet=0))
                for reaction in non_rumour.reactions:
                    tweets.append(non_rumour.source_tweet.to_json(is_rumour=1, event=event.name, is_source_tweet=1, reaction_text=reaction.text))

                    # tweets.append(reaction.to_json(is_rumour=1, event=event.name, is_source_tweet=1))

        return tweets
