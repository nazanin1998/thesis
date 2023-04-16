from tabulate import tabulate

from lib.utils import constants
from lib.models.event_model import EventModel
from lib.models.tweet_model import TweetModel
from lib.models.tweet_tree_model import TweetTreeModel
from lib.utils.file_dir_handler import FileDirHandler
from lib.training_modules.read_ds.pheme.save_pheme_csv import get_train_path_for_specified_split_size, \
    get_val_path_for_specified_split_size, get_test_path_for_specified_split_size


def print_event_summery_in_table(events):
    index = 0
    l = []

    for event in events:
        index += 1
        l.append(event.to_table_array())
    table = tabulate(l, headers=['event title', 'rumours', 'non_rumours', 'rumours', "all_non_rumours"],
                     tablefmt='orgtbl')

    print(table)


class ReadPhemeJson:
    def __init__(self):
        self.__train_path = get_train_path_for_specified_split_size()
        self.__val_path = get_val_path_for_specified_split_size()
        self.__test_path = get_test_path_for_specified_split_size()
        self.__pheme_json_dir = constants.PHEME_JSON_DIR

    def read_json(self):
        print(f"\tRead PHEME dataset Based on concate all features (.json) ... directory => {constants.PHEME_JSON_DIR}")
        events = self.__extract_events_from_json_dataset()
        print_event_summery_in_table(events)
        return events

    def __extract_events_from_json_dataset(self):
        events = []
        event_dirs = FileDirHandler.read_directories(directory=self.__pheme_json_dir)

        for event_dir in event_dirs:
            if event_dir.startswith("."):
                event_dirs.remove(event_dir)

        for event_dir in event_dirs:
            event = self.__extract_event_from_dir(event_dir=self.__pheme_json_dir + event_dir)
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
                event.non_rumors = self.tweet_trees_extraction(tweet_tree_dir=tweet_tree_dir)

            elif inner_event_dir == constants.RUMOURS:
                event.rumors = self.tweet_trees_extraction(tweet_tree_dir=tweet_tree_dir)

        return event

    @staticmethod
    def tweet_trees_extraction(tweet_tree_dir):
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
