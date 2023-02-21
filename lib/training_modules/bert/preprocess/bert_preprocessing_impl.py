from pandas import DataFrame

from lib.training_modules.bert.bert_model_name import get_bert_preprocess_model_name
from lib.training_modules.bert.preprocess.bert_preprocess_input_maker import BertPreprocessInputMaker
from lib.training_modules.bert.preprocess.bert_preprocess_layer_maker import BertPreprocessLayerMaker
from lib.training_modules.bert.preprocess.bert_preprocess_model_maker import BertPreprocessModelMaker
from lib.training_modules.bert.preprocess.bert_preprocessing import BertPreprocessing
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from lib.training_modules.bert.bert_configurations import preprocess_batch_size, preprocess_buffer_size
from lib.training_modules.bert.preprocess.bert_preprocess_ds_statistics import BertPreprocessDsStatistics
from lib.utils.log.logger import log_start_phase, log_end_phase, log_line


class BertPreprocessingImpl(BertPreprocessing):
    def __init__(
            self,
            bert_model_name='bert_en_uncased_L-12_H-768_A-12'):
        self.label_feature_name, self.categorical_feature_names, self.binary_feature_names, self.numeric_feature_names, \
        self.str_feature_names = \
            BertPreprocessDsStatistics().get_categorical_binary_numeric_string_feature_names()

        self.bert_preprocess_model_name = get_bert_preprocess_model_name(bert_model_name=bert_model_name)

    def start(self, df):
        log_start_phase(2, 'BERT PREPROCESSING')

        label_classes, x_train_tensor, x_val_tensor, x_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = \
            BertPreprocessDsStatistics().get_train_val_test_tensors(df=df)

        inputs = BertPreprocessInputMaker.make_input_for_all_ds_columns(
            df,
            self.str_feature_names,
            self.categorical_feature_names,
            self.binary_feature_names,
            self.numeric_feature_names)

        bert_preprocess = self.get_bert_preprocess_model()

        preprocess_layers = BertPreprocessLayerMaker().make_preprocess_layer_for_each_input_dtype(
            inputs,
            self.binary_feature_names,
            self.numeric_feature_names,
            self.str_feature_names,
            bert_preprocess,
            df)

        bert_pack = self.__get_bert_pack_model(bert_preprocess)

        preprocessor_model = BertPreprocessModelMaker().make_preprocess_model(
            preprocess_layers=preprocess_layers,
            inputs=inputs,
            bert_pack=bert_pack)

        print(f"Preprocess Summery")
        preprocessor_model.summary()

        train_tensor_dataset, val_tensor_dataset, test_tensor_dataset = self.preprocess_test_train_val_ds(
            preprocessor_model=preprocessor_model,
            x_train_tensor=x_train_tensor,
            x_val_tensor=x_val_tensor,
            x_test_tensor=x_test_tensor,
            y_train_tensor=y_train_tensor,
            y_val_tensor=y_val_tensor,
            y_test_tensor=y_test_tensor,
        )

        log_end_phase(2, 'BERT PREPROCESSING')
        log_line()

        return train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, label_classes, x_train_tensor.shape[0], \
               x_val_tensor.shape[0], x_test_tensor.shape[0], preprocessor_model

    @staticmethod
    def get_bert_preprocess_model():
        bert_preprocess = hub.load(get_bert_preprocess_model_name())
        return bert_preprocess

    @staticmethod
    def __get_bert_pack_model(bert_preprocess):
        bert_pack = bert_preprocess.bert_pack_inputs
        return bert_pack

    def preprocess_test_train_val_ds(self,
                                     preprocessor_model,
                                     x_train_tensor,
                                     x_val_tensor,
                                     x_test_tensor,
                                     y_train_tensor,
                                     y_val_tensor,
                                     y_test_tensor
                                     ):
        train_size = x_train_tensor.shape[0]
        val_size = x_val_tensor.shape[0]
        test_size = x_test_tensor.shape[0]
        with tf.device('/cpu:0'):
            x_train_tensor_tuple = self.__make_tuple_from_tensor(x_train_tensor)
            x_val_tensor_tuple = self.__make_tuple_from_tensor(x_val_tensor)
            x_test_tensor_tuple = self.__make_tuple_from_tensor(x_test_tensor)

            preprocessed_x_train = preprocessor_model(x_train_tensor_tuple)
            preprocessed_x_val = preprocessor_model(x_val_tensor_tuple)
            preprocessed_x_test = preprocessor_model(x_test_tensor_tuple)

        train_tensor_dataset = self.make_tensor_ds_of_preprocessed_data(
            label_tensor=y_train_tensor,
            preprocessed_train_features=preprocessed_x_train,
            num_examples=train_size,
            is_training=True)

        val_tensor_dataset = self.make_tensor_ds_of_preprocessed_data(
            label_tensor=y_val_tensor,
            preprocessed_train_features=preprocessed_x_val,
            num_examples=val_size,
            is_training=False)

        test_tensor_dataset = self.make_tensor_ds_of_preprocessed_data(
            label_tensor=y_test_tensor,
            preprocessed_train_features=preprocessed_x_test,
            num_examples=test_size,
            is_training=False)

        return train_tensor_dataset, val_tensor_dataset, test_tensor_dataset

    @staticmethod
    def __make_tuple_from_tensor(x_tensor):
        my_list = list()
        # for idx in range(0, x_tensor.shape[1]):
        #     my_list.append(x_tensor[:, idx])
        for col_name in x_tensor.keys():
            my_list.append(x_tensor[col_name])
        return tuple(my_list)

    def make_tensor_ds_of_preprocessed_data(
            self,
            label_tensor,
            preprocessed_train_features,
            num_examples,
            is_training):
        dataset = tf.data.Dataset.from_tensor_slices((preprocessed_train_features, label_tensor))
        if is_training:
            dataset = dataset.shuffle(num_examples)
            dataset = dataset.repeat()
        dataset = dataset.batch(preprocess_batch_size)
        dataset = dataset.cache().prefetch(buffer_size=preprocess_buffer_size)
        return dataset
