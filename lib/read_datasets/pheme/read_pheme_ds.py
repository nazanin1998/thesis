from lib.utils.file_dir_handler import FileDirHandler
import lib.constants as constants
from lib.read_datasets.pheme.read_pheme_csv_ds.read_pheme_csv_ds import ReadPhemeCSVDataset
from lib.read_datasets.pheme.read_pheme_json_ds.read_pheme_json_ds_impl import ReadPhemeJsonDSImpl
from lib.training_modules.bert.bert_configurations import PREPROCESS_ONLY_SOURCE_TWEET
from lib.utils.log.logger import log_line, log_start_phase, log_end_phase, log_phase_desc


def read_pheme_ds():
    pheme_csv_dirs = FileDirHandler.read_directories(directory=constants.PHEME_CSV_DIR)
    train_df, val_df, test_df = __read_ds(pheme_csv_dirs)
    __log_df_statistics(train_df, val_df, test_df)

    return train_df, val_df, test_df


def __log_df_statistics(train_df, val_df, test_df):
    log_line()
    log_start_phase(1, 'READ DATA')
    if PREPROCESS_ONLY_SOURCE_TWEET:
        log_phase_desc(f"Path (.csv) : {constants.PHEME_CSV_ONLY_TEXT_PATH}")
    else:
        log_phase_desc(
            f"Total data shape: ({train_df.shape[0] + val_df.shape[0] + test_df.shape[0]}, {test_df.shape[1]})")
        log_phase_desc(f"Path train shape: {train_df.shape}, path: {constants.PHEME_TRAIN_CSV_PATH}")
        log_phase_desc(f"Path val shape  : {val_df.shape}, path: {constants.PHEME_VAL_CSV_PATH}")
        log_phase_desc(f"Path test shape : {test_df.shape}, path: {constants.PHEME_TEST_CSV_PATH}")
    log_end_phase(1, 'READ DATA')
    log_line()


def __read_ds(pheme_csv_dirs):
    if (pheme_csv_dirs is None) or (
            not PREPROCESS_ONLY_SOURCE_TWEET and not pheme_csv_dirs.__contains__(constants.PHEME_TRAIN_CSV_NAME)) or (
            PREPROCESS_ONLY_SOURCE_TWEET and not pheme_csv_dirs.__contains__(constants.PHEME_CSV_ONLY_TEXT_NAME)):
        return ReadPhemeJsonDSImpl().read_and_save_csv()
    else:
        return ReadPhemeCSVDataset().read_csv_dataset()
