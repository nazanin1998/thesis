from abc import ABC


class BertPreprocessing(ABC):

    def start(self, df):
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

