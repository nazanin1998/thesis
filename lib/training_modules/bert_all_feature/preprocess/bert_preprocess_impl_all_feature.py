from abc import ABC

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

from lib.constants import PHEME_LABEL_COL_NAME, PHEME_LABEL_SECONDARY_COL_NAME, TRAIN, \
    VALIDATION, TEST, PHEME_PREPROCESSED_TOTAL_TEXT_COL_NAME
from lib.training_modules.bert.bert_configurations import BERT_MODEL_NAME, BERT_BATCH_SIZE, PREPROCESS_DO_SHUFFLING, \
    BERT_EPOCHS
from lib.training_modules.bert.preprocess.bert_preprocessing import BertPreprocessing


class BertPreprocessingImplAllFeatures():
    @staticmethod
    def convert_df_to_ds(df):
        return Dataset.from_pandas(df)

    def convert_df_to_ds_and_prepare_features_cols(self, df):
        df = df[[PHEME_PREPROCESSED_TOTAL_TEXT_COL_NAME, PHEME_LABEL_COL_NAME]]
        ds = self.convert_df_to_ds(df).rename_column(PHEME_LABEL_COL_NAME,
                                                     PHEME_LABEL_SECONDARY_COL_NAME)

        # ds = ds.class_encode_column(PHEME_LABEL_SECONDARY_COL_NAME)
        return ds

    def convert_splited_df_to_ds_dict(self, train_df, val_df, test_df):
        dataset = DatasetDict()

        dataset[TRAIN] = self.convert_df_to_ds_and_prepare_features_cols(train_df)
        dataset[VALIDATION] = self.convert_df_to_ds_and_prepare_features_cols(val_df)
        dataset[TEST] = self.convert_df_to_ds_and_prepare_features_cols(test_df)

        return dataset

    def __init__(self, train_df, val_df, test_df):
        self.__dataset = self.convert_splited_df_to_ds_dict(train_df, val_df, test_df)

        self.__tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def start(self):
        base_features = set(self.__dataset[TRAIN].features)

        encoded_dataset = self.__dataset.map(self.tokenizer_function, batched=True)
        tokenizer_features = list(set(encoded_dataset[TRAIN].features) - base_features)
        print("Columns added by tokenizer:", tokenizer_features)
        print(f'labels: {encoded_dataset[TRAIN].features[PHEME_LABEL_SECONDARY_COL_NAME]}')

        # print(f"tf_train_dataset {tf_train_dataset}")
        # print(f"tf_validation_dataset {tf_validation_dataset}")
        #
        # input_spec, label_spec = tf_validation_dataset.element_spec
        # print(f'input_spec {input_spec}')
        # print(f'label_spec {label_spec}')
        # input_spec, label_spec = tf_train_dataset.element_spec
        # print(f'input_spec {input_spec}')

        return encoded_dataset, self.__tokenizer

    def tokenizer_function(self, df):
        return self.__tokenizer(df[PHEME_PREPROCESSED_TOTAL_TEXT_COL_NAME],
                                padding='longest',
                                truncation=True)
