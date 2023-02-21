from lib.training_modules.bert.bert_configurations import preprocess_ignore_exc_str
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text


class BertPreprocessLayerMaker:

    def make_preprocess_layer_for_each_input_dtype(
            self,
            inputs,
            binary_feature_names,
            numeric_feature_names,
            str_feature_names,
            bert_preprocess,
            df
    ):
        preprocess_layers = []
        if not preprocess_ignore_exc_str:
            preprocess_layers = self.__make_numeric_feature_preprocess_layer(
                numeric_feature_names=numeric_feature_names,
                numeric_features=df[numeric_feature_names],
                inputs=inputs)
            preprocess_layers = self.__make_binary_feature_preprocess_layer(
                preprocess_layers=preprocess_layers,
                binary_feature_names=binary_feature_names,
                inputs=inputs)

        preprocess_layers = self.__make_string_feature_preprocess_layer(
            inputs=inputs,
            preprocess_layers=preprocess_layers,
            str_feature_names=str_feature_names,
            bert_preprocess=bert_preprocess,
            df=df)
        return preprocess_layers

    def __make_binary_feature_preprocess_layer(
            self,
            preprocess_layers,
            binary_feature_names,
            inputs):
        for name in binary_feature_names:
            inp = inputs[name]
            inp = inp[:, tf.newaxis]
            float_value = tf.cast(inp, tf.float32, )
            preprocess_layers.append(float_value)

        return preprocess_layers

    def __make_numeric_feature_preprocess_layer(
            self,
            numeric_feature_names,
            numeric_features,
            inputs):
        normalizer = self.__make_normalizer()

        normalizer.adapt(self.__stack_dict(dict(numeric_features)))

        numeric_inputs = {}
        for name in numeric_feature_names:
            numeric_inputs[name] = inputs[name]

        x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
        numeric_normalized = normalizer(x)
        preprocess_layers = [numeric_normalized]
        return preprocess_layers

    def __make_string_feature_preprocess_layer(
            self,
            inputs,
            preprocess_layers,
            str_feature_names,
            bert_preprocess,
            df
    ):

        for name, input_item in inputs.items():
            if name not in str_feature_names:
                continue
            # if bert_preprocess_do_vectorization:
            #     r"""first approach"""
            #     lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(df[name]))
            #     one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
            #     # one_hot = tf.keras.layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=None)
            #     x = lookup(input_item)
            #     preprocessed_item = one_hot(x)
            # else:
            tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name=f'tokenizer{name}')
            preprocessed_item = tokenizer(input_item)
            r"""second approach"""
            # text_vectorizer = tf.keras.layers.TextVectorization()
            #
            # text_vectorizer.adapt(df[name])
            # preprocessed_item = text_vectorizer(input_item)
            # preprocessed_item = tf.cast(preprocessed_item, dtype=tf.float32, name=f"vectorization_{name}")

            preprocess_layers.append(preprocessed_item)

        return preprocess_layers

    @staticmethod
    def __make_normalizer():
        return tf.keras.layers.Normalization(axis=-1)

    @staticmethod
    def __stack_dict(inputs, fun=tf.stack):
        values = []
        for key in sorted(inputs.keys()):
            values.append(tf.cast(inputs[key], tf.float32))

        return fun(values, axis=-1)
