from lib.constants import PHEME_LABEL_COL_NAME
from lib.training_modules.bert.preprocess.bert_preprocess_input_maker import BertPreprocessInputMaker
from lib.training_modules.bert.preprocess.bert_preprocess_model_maker import BertPreprocessModelMaker
from lib.training_modules.bert.preprocess.bert_preprocessing_impl import BertPreprocessingImpl
from lib.training_modules.bilstm.bilstm_configurations import BI_LSTM_TRAIN_SIZE, BI_LSTM_VAL_SIZE, BI_LSTM_TEST_SIZE
from lib.utils.log.logger import log_start_phase, log_phase_desc, print_indented_key_value, log_end_phase, log_line
import tensorflow as tf


class BiLstmPreprocess:
    def __init__(self):
        self.__label_feature_name = PHEME_LABEL_COL_NAME
        self.__training_feature_names = ['text', ]

    def __get_x_y_from_df(self, df):
        y = df[self.__label_feature_name]
        x = df[self.__training_feature_names]
        return x, y

    def start(self, train_df, val_df, test_df):
        log_start_phase(2, 'BI-LSTM PREPROCESSING')

        x_train, y_train = self.__get_x_y_from_df(train_df)
        x_test, y_test = self.__get_x_y_from_df(test_df)
        x_val, y_val = self.__get_x_y_from_df(val_df)

        log_phase_desc(f'TRAINING FEATURE  : {self.__training_feature_names}')
        log_phase_desc(f'LABEL FEATURE     : {self.__label_feature_name}\n')

        log_phase_desc(f'TRAIN      (SIZE) : {x_train.shape} ({BI_LSTM_TRAIN_SIZE * 100}%)')
        log_phase_desc(f'VALIDATION (SIZE) : {x_val.shape} ({BI_LSTM_VAL_SIZE * 100}%)')
        log_phase_desc(f'TEST       (SIZE) : {x_test.shape} ({BI_LSTM_TEST_SIZE * 100}%)')
        print_indented_key_value(f'\tYTrain classes\t  : ', f'{y_train.value_counts()}', intend_num=6)
        print_indented_key_value(f'\tYVal classes\t  : ', f'{y_val.value_counts()}', intend_num=6)
        print_indented_key_value(f'\tYTest classes\t  : ', f'{y_test.value_counts()}', intend_num=6)

        inputs = BertPreprocessInputMaker.make_input_for_all_ds_columns(
            x_train,
            self.__training_feature_names,
            None,
            None,
            None)

        preprocess_layers = []
        preprocess_layers = self.__make_string_feature_preprocess_layer(
            inputs,
            preprocess_layers,
            self.__training_feature_names,
            x_train)

        preprocessor_model = self.__get_bi_lstm_preprocess_model(preprocess_layers, inputs)

        print(f"Preprocess Summery")
        preprocessor_model.summary()

        train_tensor_dataset, val_tensor_dataset, test_tensor_dataset = BertPreprocessingImpl() \
            .preprocess_test_train_val_ds(
            preprocessor_model=preprocessor_model,
            x_train_tensor=x_train,
            x_val_tensor=x_val,
            x_test_tensor=x_test,
            y_train_tensor=y_train,
            y_val_tensor=y_val,
            y_test_tensor=y_test,
        )

        log_end_phase(2, 'BI_LSTM PREPROCESSING')
        log_line()

        return train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, x_train.shape[0], \
               x_val.shape[0], x_test.shape[0], preprocessor_model

    @staticmethod
    def __make_string_feature_preprocess_layer(
            inputs,
            preprocess_layers,
            str_feature_names,
            x_train
    ):

        for name, input_item in inputs.items():
            if name not in str_feature_names:
                continue

            encoder = tf.keras.layers.TextVectorization()

            encoder.adapt(x_train[name])
            # encoder.adapt(x_train.map(lambda text, label: text))

            preprocessed_item = encoder(input_item)
            preprocessed_item = tf.cast(preprocessed_item, dtype=tf.float32, name=f"vectorization_{name}")
            preprocess_layers.append(preprocessed_item)

        return preprocess_layers

    @staticmethod
    def __get_bi_lstm_preprocess_model(preprocess_layers, inputs):

        preprocessed_result = tf.concat(preprocess_layers, axis=-1)
        # preprocessed_result = tf.keras.layers.Concatenate()(preprocess_layers)

        preprocessor = tf.keras.Model(inputs, preprocessed_result)
        BertPreprocessModelMaker.plot_preprocess_model(preprocessor)
        return preprocessor
