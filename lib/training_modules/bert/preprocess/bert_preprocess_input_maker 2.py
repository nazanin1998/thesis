import tensorflow as tf

from lib.training_modules.bert.bert_configurations import PREPROCESS_IGNORE_EXC_STR


class BertPreprocessInputMaker:

    # string dtype = string
    # int dtype = float32
    # categorical and binary dtype = int64
    @staticmethod
    def make_input_for_all_ds_columns(
            x_train,
            str_feature_names,
            categorical_feature_names,
            binary_feature_names,
            numeric_feature_names,
            input_shape=()
    ):
        inputs = {}

        for name, column in x_train.items():

            if name in str_feature_names:
                d_type = tf.string
                inputs[name] = tf.keras.Input(shape=input_shape, name=name, dtype=d_type)
                continue
            if PREPROCESS_IGNORE_EXC_STR:
                continue
            if (name in categorical_feature_names or
                    name in binary_feature_names):
                d_type = tf.int32
            elif name in numeric_feature_names:
                d_type = tf.float32
            else:
                continue
            inputs[name] = tf.keras.Input(shape=input_shape, name=name, dtype=d_type)

        return inputs
