import lib.constants as constants
import pandas as pd

from lib.read_datasets.pheme.file_dir_handler import FileDirHandler
from lib.training_modules.bert.bert_configurations import only_source_tweet


class ReadPhemeCSVDataset:
    def __init__(self):
        self.df = None

    def read_csv_dataset(self):
        print("\tRead dataset (.csv) ...")

        if only_source_tweet:
            self.df = FileDirHandler.read_csv_file(path=constants.PHEME_CSV_ONLY_TEXT_PATH)
        else:
            self.df = FileDirHandler.read_csv_file(path=constants.PHEME_CSV_PATH)

        return self.df
