import lib.constants as constants

from lib.utils.file_dir_handler import FileDirHandler
from lib.training_modules.bert.bert_configurations import PREPROCESS_ONLY_SOURCE_TWEET


class ReadPhemeCSVDataset:

    def read_csv_dataset(self):
        print("\tRead dataset (.csv) ...")

        if PREPROCESS_ONLY_SOURCE_TWEET:
            self.train_df = FileDirHandler.read_csv_file(path=constants.PHEME_CSV_ONLY_TEXT_PATH)
        else:
            train_df = FileDirHandler.read_csv_file(path=constants.PHEME_TRAIN_CSV_PATH)
            val_df = FileDirHandler.read_csv_file(path=constants.PHEME_VAL_CSV_PATH)
            test_df = FileDirHandler.read_csv_file(path=constants.PHEME_TEST_CSV_PATH)
            return train_df, val_df, test_df
