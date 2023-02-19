from abc import ABC


class BertPreprocessing(ABC):

    def start(self, df):
        pass

    def make_binary_feature_preprocess_layer(
            self,
            preprocess_layers,
            binary_feature_names,
            inputs):
        pass

    def make_numeric_feature_preprocess_layer(
            self,
            preprocess_layers,
            numeric_feature_names,
            numeric_features,
            inputs):
        pass

    def make_string_feature_preprocess_layer(
            self,
            inputs,
            preprocess_layers,
            str_feature_names, df,input_shape):
        pass

    def make_preprocess_model(
            self,
            preprocess_layers,
            inputs,
            bert_pack):
        pass

    def make_tensor_ds_of_preprocessed_data(
            self,
            label_tensor,
            preprocessed_train_features,
            num_examples,
            is_training):
        pass

