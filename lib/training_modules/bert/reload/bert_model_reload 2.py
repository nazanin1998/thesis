import tensorflow as tf

from lib.training_modules.bert.analysis.bert_model_analysis import BertModelAnalysis
from lib.training_modules.bert.train.bert_model_impl import BertModelImpl


def prepare_serving(record):
    model_inputs = {ft: record[ft] for ft in ['text', "reaction_text"]}
    return model_inputs


class BertModelReload:
    def __init__(self, test_tensor_dataset):
        saved_model_path = BertModelImpl.get_save_model_path()
        print(f'saved path {saved_model_path}')
        reloaded_model = tf.saved_model.load(saved_model_path)
        print(reloaded_model)
        BertModelAnalysis(model=reloaded_model, history=None).plot_bert_model()
        # BertModelAnalysis(model=reloaded_model, history=None).evaluation()
        #
        # print( reloaded_model.signatures)
        # print( test_tensor_dataset.shuffle(1000))
        # print( test_tensor_dataset.shuffle(1000))
        serving_model = reloaded_model.signatures['serving_default']
        serving_model(test_tensor_dataset)
        for test_row in test_tensor_dataset.shuffle(1000).map(prepare_serving).take(5):
            result = serving_model(**test_row)
            print(f"prediction { result['prediction']}")
            print(f"list(test_row.values()) {list(test_row.values())}")
            # The 'prediction' key is the classifier's defined model name.
            # print_bert_results(list(test_row.values()), result['prediction'], tfds_name)

        # test_loss, test_accuracy = reloaded_model(test_tensor_dataset)
        # print(test_loss)
        # print(test_accuracy)
