from datasets import DatasetDict
from transformers import AutoTokenizer

from lib.training_modules.base.preprocess.base_preprocess import convert_df_to_ds, merge_3_dataframes
from lib.utils.constants import PHEME_LABEL_COL_NAME, PHEME_LABEL_SECONDARY_COL_NAME, TRAIN, \
    VALIDATION, TEST, PHEME_TOTAL_TEXT_SECONDARY_COL_NAME
from lib.training_modules.bert.bert_configurations import BERT_MODEL_NAME, BERT_USE_K_FOLD


class BertPreprocessing:
    def __init__(self, train_df, val_df, test_df):
        self.__dataset = self.__convert_splitting_df_to_ds_dict(train_df, val_df, test_df)
        self.__tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def convert_df_to_ds_and_prepare_features_cols(self, df):
        df = df[[PHEME_TOTAL_TEXT_SECONDARY_COL_NAME, PHEME_LABEL_COL_NAME]]
        ds = convert_df_to_ds(df)
        ds = ds.rename_column(PHEME_LABEL_COL_NAME, PHEME_LABEL_SECONDARY_COL_NAME)

        # ds = ds.class_encode_column(PHEME_LABEL_SECONDARY_COL_NAME)
        return ds

    def __convert_splitting_df_to_ds_dict(self, train_df, val_df, test_df):
        dataset = DatasetDict()

        if BERT_USE_K_FOLD:
            df = merge_3_dataframes(train_df, val_df, test_df)
            dataset[TRAIN] = self.convert_df_to_ds_and_prepare_features_cols(df)
        else:
            dataset[TRAIN] = self.convert_df_to_ds_and_prepare_features_cols(train_df)
            dataset[VALIDATION] = self.convert_df_to_ds_and_prepare_features_cols(val_df)
            dataset[TEST] = self.convert_df_to_ds_and_prepare_features_cols(test_df)

        return dataset

    def start(self):
        base_features = set(self.__dataset[TRAIN].features)

        encoded_dataset = self.__dataset.map(self.tokenizer_function, batched=True)
        tokenizer_features = list(set(encoded_dataset[TRAIN].features) - base_features)

        return encoded_dataset, self.__tokenizer

    def tokenizer_function(self, df):
        return self.__tokenizer(df[PHEME_TOTAL_TEXT_SECONDARY_COL_NAME],
                                padding='longest',
                                truncation=True)

