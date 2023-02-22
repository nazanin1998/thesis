import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_text as text  # A dependency of the preprocessing model

# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
# https://www.tensorflow.org/tutorials/load_data/csv


ignore_exc_str = True


def convert_df_to_tensor_ds(df):
    seq_length = 32

    label_feature_name, categorical_feature_names, binary_feature_names, numeric_feature_names, str_feature_names = \
        __get_categorical_binary_numeric_string_feature_names()

    available_splits, num_examples, num_classes = __ds_statistics(
        df=df,
        label_feature_name=label_feature_name,
        categorical_feature_names=categorical_feature_names,
        binary_feature_names=binary_feature_names,
        numeric_feature_names=numeric_feature_names,
        str_feature_names=str_feature_names)

    inputs = __make_input_for_all_dtypes(
        df=df,
        str_feature_names=str_feature_names,
        binary_feature_names=binary_feature_names,
        categorical_feature_names=categorical_feature_names,
        ignore_exc_str=ignore_exc_str)

    print(f'CONVERT (Dataset2Tensor) => InputShape: {df.shape}')
    print(f'INPUT FOR ALL DATA_TYPES: {inputs}')

    preprocess_layers = []

    if not ignore_exc_str:
        preprocess_layers = _binary_feature_df_to_tensor(
            preprocess_layers=preprocess_layers,
            binary_feature_names=binary_feature_names,
            inputs=inputs)

        preprocess_layers = __normalize_numeric_features(
            preprocess_layers=preprocess_layers,
            numeric_feature_names=numeric_feature_names,
            numeric_features=df[numeric_feature_names],
            inputs=inputs)

    preprocess_layers, bert_pack = __str_feature_preprocess_layer(
        inputs=inputs,
        preprocess_layers=preprocess_layers,
        str_feature_names=str_feature_names)

    preprocessor_model = __make_preprocess_model(
        preprocess_layers=preprocess_layers,
        inputs=inputs,
        bert_pack=bert_pack,
        seq_length=seq_length)

    print(preprocessor_model.summary())

    train_features = df[str_feature_names]
    label_feature = df[label_feature_name]

    train_features_tensor = _convert_to_tensor(train_features, dtype=tf.string)
    label_feature_tensor = _convert_to_tensor(label_feature, dtype=tf.int64)

    with tf.device('/cpu:0'):
        preprocessed_train_features = preprocessor_model(train_features_tensor)

    print(f'TRAIN_FEATURES            (SHAPE) : {train_features.shape}')
    print(f'LABELS                    (SHAPE) : {label_feature.shape}')
    print(f'PREPROCESSED TRAIN_TENSOR : {preprocessed_train_features}')

    train_dataset, train_data_size = load_dataset_from_tfds(
        df=df,
        preprocessed_train_features=preprocessed_train_features,
        num_examples=num_examples,
        batch_size=seq_length,
        is_training=True)
    return train_dataset, train_data_size, num_classes


def __ds_statistics(df,
                    label_feature_name,
                    categorical_feature_names,
                    binary_feature_names,
                    numeric_feature_names,
                    str_feature_names):
    available_splits = list(['train', 'validation', 'test'])
    num_classes = len(df['is_rumour'].value_counts())
    num_examples = df.shape[0]

    print(f'PHEME DS (SHAPE)      : {df.shape}')
    print(f'CLASSIFICATION CLASSES: {num_classes}')
    print(f'TRAINING FEATURE : {str_feature_names}\n')

    return available_splits, num_examples, num_classes


def load_dataset_from_tfds(df, preprocessed_train_features, num_examples, batch_size,
                           is_training):
    dataset = tf.data.Dataset.from_tensor_slices((preprocessed_train_features, df['is_rumour']))
    print(f'ds is: {dataset}')

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    # dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    print(f'ds is: {dataset}')
    return dataset, num_examples


def __str_feature_preprocess_layer(inputs,
                                   preprocess_layers,
                                   str_feature_names):
    bert_preprocess = hub.load(bert_preprocess_model_name)

    for name, input_item in inputs.items():
        if name not in str_feature_names:
            continue

        tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')

        preprocessed_item = tokenizer(input_item)

        r"""first approach"""
        # lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(df[name]))
        # one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
        # one_hot = tf.keras.layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=None)

        # x = lookup(input_item)
        # preprocessed_item = one_hot(x)

        r"""second approach"""
        # text_vectorizer = tf.keras.layers.TextVectorization()
        # text_vectorizer.adapt(df[name])
        #
        # preprocessed_item = text_vectorizer(input_item)

        preprocess_layers.append(preprocessed_item)

    bert_pack = bert_preprocess.bert_pack_inputs
    return preprocess_layers, bert_pack


def __make_preprocess_model(preprocess_layers, inputs, bert_pack, seq_length):
    packer = hub.KerasLayer(bert_pack,
                            # arguments=dict(seq_length=seq_length),
                            name='packer')
    preprocessed_result = packer(preprocess_layers)
    # preprocessed_result = tf.concat(preprocess_layers, axis=-1)

    preprocessor = tf.keras.Model(inputs, preprocessed_result)
    tf.keras.utils.plot_model(preprocessor,
                              rankdir="LR",
                              # show_shapes=True,
                              to_file='preprocess_layers.png')
    return preprocessor


def __make_normalizer():
    return tf.keras.layers.Normalization(axis=-1)


def __normalize_numeric_features(preprocess_layers,
                                 numeric_feature_names,
                                 numeric_features,
                                 inputs):
    normalizer = __make_normalizer()

    normalizer.adapt(stack_dict(dict(numeric_features)))

    numeric_inputs = {}
    for name in numeric_feature_names:
        numeric_inputs[name] = inputs[name]

    numeric_inputs = stack_dict(numeric_inputs)
    numeric_normalized = normalizer(numeric_inputs)

    preprocess_layers.append(numeric_normalized)
    return preprocess_layers


def _convert_to_tensor(feature, dtype=None):
    return tf.convert_to_tensor(feature, dtype=dtype)


def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
        values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


def __make_input_for_all_dtypes(df,
                                str_feature_names,
                                categorical_feature_names,
                                binary_feature_names,
                                ignore_exc_str=False):
    inputs = {}
    for name, column in df.items():

        if ignore_exc_str and name in str_feature_names:
            d_type = tf.string
            inputs[name] = tf.keras.Input(shape=(), name=name, dtype=d_type)

        if ignore_exc_str:
            continue

        if (name in categorical_feature_names or
                name in binary_feature_names):
            d_type = tf.int64
        else:
            d_type = tf.float32

        inputs[name] = tf.keras.Input(shape=(), name=name, dtype=d_type)

    return inputs


def _binary_feature_df_to_tensor(preprocess_layers, binary_feature_names, inputs):
    for name in binary_feature_names:
        inp = inputs[name]
        inp = inp[:, tf.newaxis]
        float_value = tf.cast(inp, tf.float32)
        preprocess_layers.append(float_value)

    return preprocess_layers


def __get_categorical_binary_numeric_string_feature_names():
    categorical_feature_names = []
    str_feature_names = ['text']
    binary_feature_names = ['is_truncated', 'is_source_tweet', 'user.verified', 'user.protected', ]
    numeric_feature_names = ['tweet_id', 'tweet_length', 'symbol_count', 'mentions_count', 'urls_count',
                             'retweet_count', 'favorite_count', 'hashtags_count', 'in_reply_user_id',
                             'in_reply_tweet_id', 'user.id', 'user.name_length', 'user.listed_count',
                             'user.tweets_count', 'user.statuses_count', 'user.friends_count',
                             'user.favourites_count', 'user.followers_count', 'user.follow_request_sent', ]

    label_feature_name = 'is_rumour'
    return label_feature_name, categorical_feature_names, binary_feature_names, numeric_feature_names, str_feature_names
