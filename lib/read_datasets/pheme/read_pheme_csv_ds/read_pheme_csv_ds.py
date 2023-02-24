import lib.constants as constants
from lib.read_datasets.pheme.read_pheme_json_ds.read_pheme_json_ds_impl import get_train_path_for_specified_split_size, \
    get_val_path_for_specified_split_size, get_test_path_for_specified_split_size

from lib.utils.file_dir_handler import FileDirHandler
from lib.training_modules.bert.bert_configurations import PREPROCESS_ONLY_SOURCE_TWEET


class ReadPhemeCSVDataset:

    def read_csv_dataset(self):
        print("\tRead dataset (.csv) ...")

        if PREPROCESS_ONLY_SOURCE_TWEET:
            self.train_df = FileDirHandler.read_csv_file(path=constants.PHEME_CSV_ONLY_TEXT_PATH)
        else:
            train_df = FileDirHandler.read_csv_file(path=get_train_path_for_specified_split_size())
            val_df = FileDirHandler.read_csv_file(path=get_val_path_for_specified_split_size())
            test_df = FileDirHandler.read_csv_file(path=get_test_path_for_specified_split_size())
            return train_df, val_df, test_df
