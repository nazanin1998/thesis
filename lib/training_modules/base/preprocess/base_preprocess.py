from datasets import Dataset
from sklearn.model_selection import train_test_split

from lib.training_modules.bert.bert_configurations import PREPROCESS_DO_SHUFFLING


def convert_df_to_ds(df):
    return Dataset.from_pandas(df)


def merge_3_dataframes(ds1, ds2, ds3):
    df = ds1[:]
    df = df.append(ds2)
    df = df.append(ds3)
    return df


def my_train_val_test_split(x, y, train_size, val_size, test_size):
    x_train, x_test_val, y_train, y_test_val = train_test_split(x, y,
                                                                train_size=train_size,                                                                shuffle=PREPROCESS_DO_SHUFFLING,
                                                                stratify=y)
    relative_val_size = val_size / (val_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val,
                                                    train_size=relative_val_size,
                                                    test_size=1 - relative_val_size,
                                                    shuffle=PREPROCESS_DO_SHUFFLING,
                                                    stratify=y_test_val,
                                                    )
    return x_train, x_val, x_test, y_train, y_val, y_test