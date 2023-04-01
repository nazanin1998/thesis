from lib.training_modules.basic_reading_ds.file_dir_handler import FileDirHandler
import lib.constants as constants
from lib.training_modules.bert.bert_configurations import BERT_TRAIN_SIZE, BERT_VAL_SIZE, BERT_TEST_SIZE
from lib.training_modules.read_ds.pheme.read_pheme_csv import ReadPhemeCsv
from lib.training_modules.read_ds.pheme.read_pheme_json import ReadPhemeJson
from lib.training_modules.read_ds.pheme.save_pheme_csv import SavePhemeCsv, get_directory_for_specified_split_size, \
    get_train_path_for_specified_split_size, get_val_path_for_specified_split_size, \
    get_test_path_for_specified_split_size
from lib.utils.log.logger import log_line, log_start_phase, log_end_phase, log_phase_desc


def read_pheme():
    train_df, val_df, test_df = __read_csv_or_json()
    __log_df_statistics(train_df, val_df, test_df)

    return train_df, val_df, test_df


def get_pheme_csv_dirs():
    return FileDirHandler.read_directories(directory=get_directory_for_specified_split_size())


def __log_df_statistics(train_df, val_df, test_df):
    log_line()
    log_start_phase(1, 'READ DATA')

    log_phase_desc(
        f"Total data shape: ({train_df.shape[0] + val_df.shape[0] + test_df.shape[0]}, {test_df.shape[1]})")
    log_phase_desc(f"Path train shape: {train_df.shape}, path: {get_train_path_for_specified_split_size()}")
    log_phase_desc(f"Path val shape  : {val_df.shape}, path: {get_val_path_for_specified_split_size()}")
    log_phase_desc(f"Path test shape : {test_df.shape}, path: {get_test_path_for_specified_split_size()}")

    log_end_phase(1, 'READ DATA')
    log_line()


def __read_csv_or_json():
    pheme_csv_dirs = get_pheme_csv_dirs()
    if (pheme_csv_dirs is None) or (not pheme_csv_dirs.__contains__(constants.PHEME_TRAIN_CSV_NAME)):
        events = ReadPhemeJson().read_json()
        return SavePhemeCsv().extract_csv_from_events(events)
    else:
        return ReadPhemeCsv().read_csv()
