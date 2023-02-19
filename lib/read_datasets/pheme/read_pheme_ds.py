from lib.read_datasets.pheme.file_dir_handler import FileDirHandler
import lib.constants as constants
from lib.read_datasets.pheme.read_pheme_csv_dataset import ReadPhemeCSVDataset
from lib.read_datasets.pheme.read_pheme_json_dataset import ReadPhemeJsonDataset
from lib.utils.log.logger import log_line, log_start_phase, log_end_phase, log_phase_desc


def read_pheme_ds():
    pheme_csv_dirs = FileDirHandler.read_directories(directory=constants.PHEME_CSV_DIR)
    df = __read_ds(pheme_csv_dirs)
    __log_df_statistics(df)

    return df


def __log_df_statistics(df):
    log_line()
    log_start_phase(1, 'READ DATA')
    log_phase_desc("Path (.csv) : " + constants.PHEME_CSV_PATH)
    log_phase_desc("Shape (.csv) : " + str(df.shape))
    log_end_phase(1, 'READ DATA')
    log_line()


def __read_ds(pheme_csv_dirs):
    if pheme_csv_dirs is None or not pheme_csv_dirs.__contains__(constants.PHEME_CSV_NAME):
        return ReadPhemeJsonDataset().read_and_save_csv()
    else:
        return ReadPhemeCSVDataset().read_csv_dataset()
