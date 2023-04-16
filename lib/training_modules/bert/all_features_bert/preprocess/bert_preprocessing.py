from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from lib.utils.constants import PHEME_LABEL_COL_NAME, PHEME_LABEL_SECONDARY_COL_NAME, TRAIN, \
    VALIDATION, TEST, PHEME_TOTAL_TEXT_SECONDARY_COL_NAME
from lib.training_modules.bert.bert_configurations import BERT_MODEL_NAME
from lib.training_modules.bert_all_feature.preprocess.basic_preprocessing import BasicPreprocessing


class BertPreprocessing:
    def __init__(self, train_df, val_df, test_df):
        self.__basic_preprocess = BasicPreprocessing()
        self.__dataset = self.convert_splited_df_to_ds_dict(train_df, val_df, test_df)
        self.__tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def convert_df_to_ds_and_prepare_features_cols(self, df):
        df = df[[PHEME_TOTAL_TEXT_SECONDARY_COL_NAME, PHEME_LABEL_COL_NAME]]
        ds = self.__basic_preprocess.convert_df_to_ds(df)
        ds = ds.rename_column(PHEME_LABEL_COL_NAME, PHEME_LABEL_SECONDARY_COL_NAME)

        # ds = ds.class_encode_column(PHEME_LABEL_SECONDARY_COL_NAME)
        return ds

    def convert_splited_df_to_ds_dict(self, train_df, val_df, test_df):
        dataset = DatasetDict()
        dataset[TRAIN] = self.convert_df_to_ds_and_prepare_features_cols(train_df)
        dataset[VALIDATION] = self.convert_df_to_ds_and_prepare_features_cols(val_df)
        dataset[TEST] = self.convert_df_to_ds_and_prepare_features_cols(test_df)

        return dataset

    def start(self):
        base_features = set(self.__dataset[TRAIN].features)

        encoded_dataset = self.__dataset.map(self.tokenizer_function, batched=True)
        tokenizer_features = list(set(encoded_dataset[TRAIN].features) - base_features)

        print("Columns added by tokenizer:", tokenizer_features)
        print(f'labels: {encoded_dataset[TRAIN].features[PHEME_LABEL_SECONDARY_COL_NAME]}')

        return encoded_dataset, self.__tokenizer

    def tokenizer_function(self, df):
        return self.__tokenizer(df[PHEME_TOTAL_TEXT_SECONDARY_COL_NAME],
                                padding='longest',
                                truncation=True)
