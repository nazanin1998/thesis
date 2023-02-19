import tensorflow as tf

from lib.training_modules.bert.bert_configurations import preprocess_ignore_exc_str


class BertInputMaker:
    # string dtype = string
    # int dtype = float32
    # categorical and binary dtype = int64
    @staticmethod
    def make_input_for_all_ds_columns(
            df,
            str_feature_names,
            categorical_feature_names,
            binary_feature_names,
            input_shape=()
    ):
        inputs = {}

        for name, column in df.items():

            if preprocess_ignore_exc_str and name in str_feature_names:
                d_type = tf.string
                inputs[name] = tf.keras.Input(shape=input_shape, name=name, dtype=d_type)

            if preprocess_ignore_exc_str:
                continue

            if (name in categorical_feature_names or
                    name in binary_feature_names):
                d_type = tf.int64
            else:
                d_type = tf.float32

            inputs[name] = tf.keras.Input(shape=input_shape, name=name, dtype=d_type)

        return inputs
