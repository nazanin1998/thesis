from lib.training_modules.bert.bert_configurations import PREPROCESS_SEQ_LEN
import tensorflow_hub as hub
import tensorflow as tf


class BertPreprocessModelMaker:

    def make_preprocess_model(
            self,
            preprocess_layers,
            inputs,
            bert_pack):
        packer = hub.KerasLayer(bert_pack,
                                arguments=dict(seq_length=PREPROCESS_SEQ_LEN),
                                name='packer')

        preprocessed_result = packer(preprocess_layers)
        # preprocessed_result = tf.concat(preprocess_layers, axis=-1)
        # preprocessed_result = tf.keras.layers.Concatenate()(preprocess_layers)

        preprocessor = tf.keras.Model(inputs, preprocessed_result)
        self.plot_preprocess_model(preprocessor)
        return preprocessor

    def plot_preprocess_model(self, model, img_name='preprocess_model.png', show_shapes=True):
        tf.keras.utils.plot_model(model,
                                  rankdir="LR",
                                  # show_shapes=show_shapes,
                                  to_file=img_name)
