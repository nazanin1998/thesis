from lib.utils.file_dir_handler import FileDirHandler
from lib.training_modules.bert.bert_configurations import BERT_USE_K_FOLD
from lib.dataset_repositories.pheme.save_pheme_csv import get_train_path_for_specified_split_size, \
    get_val_path_for_specified_split_size, get_test_path_for_specified_split_size


class ReadPhemeCsv:

    def __init__(self):
        self.__train_path = get_train_path_for_specified_split_size()
        self.__val_path = get_val_path_for_specified_split_size()
        self.__test_path = get_test_path_for_specified_split_size()

    def read_csv(self):
        print("\tRead dataset (.csv) ...")
        train_df = FileDirHandler.read_csv_file(path=self.__train_path)
        val_df = FileDirHandler.read_csv_file(path=self.__val_path)
        test_df = FileDirHandler.read_csv_file(path=self.__test_path)
        return train_df, val_df, test_df
