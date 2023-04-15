from sklearn.model_selection import train_test_split

from lib.training_modules.bert.bert_configurations import PREPROCESS_DO_SHUFFLING


def my_train_val_test_split(x, y, train_size, val_size, test_size):
    x_train, x_test_val, y_train, y_test_val = train_test_split(x, y,
                                                                train_size=train_size,
                                                                shuffle=PREPROCESS_DO_SHUFFLING,
                                                                stratify=y)

    relative_val_size = val_size / (val_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val,
                                                    train_size=relative_val_size,
                                                    test_size=1 - relative_val_size,
                                                    shuffle=PREPROCESS_DO_SHUFFLING,
                                                    stratify=y_test_val,
                                                    )
    return x_train, x_val, x_test, y_train, y_val, y_test