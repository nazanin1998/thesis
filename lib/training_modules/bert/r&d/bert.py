from abc import ABC


class Bert(ABC):
    def get_test_train_ds(self,
                          df,
                          label_name='is_rumour',
                          col_name='text_pre',
                          test_size=0.2):
        pass

    def ds_review(self, ds, class_names):
        pass

    def get_bert_preprocess_model(self, model_name):
        pass

    def get_bert_model(self, model_name):
        pass

    def do_preprocess(self, test_str_arr, preprocess_model):
        pass

    def do_bert(self, preprocessed_text, bert_model):
        pass

