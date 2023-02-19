import os

import numpy
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text  # A dependency of the preprocessing model
import tensorflow_addons as tfa
import numpy as np
from keras.applications.densenet import layers
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from lib.training_modules.bert.bert_model_name import get_bert_model_name, get_bert_preprocess_model_name


def __bert_tokenize(item, tokenizer):
    print(item)
    if item is None or item is numpy.NaN or len(item) == 0:
        return []
    b = tokenizer(tf.constant([item]), )

    print(b)
    return b


bert_model_name = get_bert_model_name()
bert_preprocess_model_name = get_bert_preprocess_model_name()


def __ds_statistics(df):
    tfds_name = 'PHEME'

    sentence_features = list(['text_pre'])

    available_splits = list(['train', 'validation', 'test'])
    num_classes = len(df['is_rumour'].value_counts())
    num_examples = df.shape[0]

    print(f'Using {tfds_name} from TFDS')
    print(f'This dataset has {num_examples} examples')
    print(f'Number of classes: {num_classes}')
    print(f'Features {sentence_features}')
    print(f'Splits {available_splits}')

    return sentence_features, available_splits, num_examples, num_classes


def run_my_bert(df, dicted_ds):
    sentence_features, available_splits, num_examples, num_classes = __ds_statistics(df)

    # with tf.device('/job:localhost'):
    #     in_memory_ds = tfds.load('glue/cola', batch_size=-1, shuffle_files=True)
    # print(in_memory_ds['train'])
    # print(type(in_memory_ds['train']))

    epochs = 1
    batch_size = 32
    init_lr = 2e-5

    print(f'Fine tuning {bert_model_name} model')
    bert_preprocess_model = make_bert_preprocess_model(sentence_features)

    # with strategy.scope():
    # metric have to be created inside the strategy scope
    # metrics, loss = get_configuration(tfds_name)

    # train_dataset, train_data_size = \
    load_dataset_from_phemeds(df, batch_size, bert_preprocess_model)
    # steps_per_epoch = train_data_size // batch_size
    # num_train_steps = steps_per_epoch * epochs
    # num_warmup_steps = num_train_steps // 10
    return
    alidation_dataset, validation_data_size = load_dataset_from_tfds(
        in_memory_ds, tfds_info, validation_split, batch_size,
        bert_preprocess_model)
    validation_steps = validation_data_size // batch_size

    classifier_model = build_classifier_model(num_classes)

    # optimizer = transformers.create_optimizer(
    #     init_lr=init_lr,
    #     num_train_steps=num_train_steps,
    #     num_warmup_steps=num_warmup_steps,
    #     # optimizer_type='adamw',
    # )

    classifier_model.compile(
        # optimizer=optimizer,
        loss=loss, metrics=[metrics])

    classifier_model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps)
    main_save_path = './my_models'
    bert_type = bert_model_name.split('/')[-2]
    saved_model_name = f'{tfds_name.replace("/", "_")}_{bert_type}'

    saved_model_path = os.path.join(main_save_path, saved_model_name)

    preprocess_inputs = bert_preprocess_model.inputs
    bert_encoder_inputs = bert_preprocess_model(preprocess_inputs)
    bert_outputs = classifier_model(bert_encoder_inputs)
    model_for_export = tf.keras.Model(preprocess_inputs, bert_outputs)

    print('Saving', saved_model_path)

    # Save everything on the Colab host (even the variables from TPU memory)
    save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    model_for_export.save(saved_model_path, include_optimizer=False,
                          options=save_options)


def run_bert(df):
    tfds_name = 'glue/cola'

    tfds_info = tfds.builder(tfds_name).info

    sentence_features = list(tfds_info.features.keys())
    sentence_features.remove('idx')
    sentence_features.remove('label')

    available_splits = list(tfds_info.splits.keys())
    train_split = 'train'
    validation_split = 'validation'
    test_split = 'test'
    if tfds_name == 'glue/mnli':
        validation_split = 'validation_matched'
        test_split = 'test_matched'

    num_classes = tfds_info.features['label'].num_classes
    num_examples = tfds_info.splits.total_num_examples

    print(f'Using {tfds_name} from TFDS')
    print(f'This dataset has {num_examples} examples')
    print(f'Number of classes: {num_classes}')
    print(f'Features {sentence_features}')
    print(f'Splits {available_splits}')

    with tf.device('/job:localhost'):
        # batch_size=-1 is a way to load the dataset into memory
        in_memory_ds = tfds.load(tfds_name, batch_size=-1, shuffle_files=True)
    print(in_memory_ds)

    epochs = 1
    batch_size = 32
    init_lr = 2e-5

    print(f'Fine tuning {bert_model_name} model')
    bert_preprocess_model = make_bert_preprocess_model(sentence_features)

    # with strategy.scope():
    # metric have to be created inside the strategy scope
    metrics, loss = get_configuration(tfds_name)

    train_dataset, train_data_size = load_dataset_from_tfds(
        in_memory_ds, tfds_info, train_split, batch_size, bert_preprocess_model)
    steps_per_epoch = train_data_size // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = num_train_steps // 10

    validation_dataset, validation_data_size = load_dataset_from_tfds(
        in_memory_ds, tfds_info, validation_split, batch_size,
        bert_preprocess_model)
    validation_steps = validation_data_size // batch_size

    classifier_model = build_classifier_model(num_classes)

    # optimizer = transformers.create_optimizer(
    #     init_lr=init_lr,
    #     num_train_steps=num_train_steps,
    #     num_warmup_steps=num_warmup_steps,
    #     # optimizer_type='adamw',
    # )

    classifier_model.compile(
        # optimizer=optimizer,
        loss=loss, metrics=[metrics])

    classifier_model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=validation_steps)
    main_save_path = './my_models'
    bert_type = bert_model_name.split('/')[-2]
    saved_model_name = f'{tfds_name.replace("/", "_")}_{bert_type}'

    saved_model_path = os.path.join(main_save_path, saved_model_name)

    preprocess_inputs = bert_preprocess_model.inputs
    bert_encoder_inputs = bert_preprocess_model(preprocess_inputs)
    bert_outputs = classifier_model(bert_encoder_inputs)
    model_for_export = tf.keras.Model(preprocess_inputs, bert_outputs)

    print('Saving', saved_model_path)

    # Save everything on the Colab host (even the variables from TPU memory)
    save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    model_for_export.save(saved_model_path, include_optimizer=False,
                          options=save_options)


def run_bert_old(df):
    r"""split test and train"""

    train_df = df['text_pre'][:200]
    test_df = df['text_pre'][201:300]
    train_label = df['is_rumour'][:200]
    test_label = df['text_pre'][201:300]
    print('train_df.shape: ' + str(train_df.shape))
    print('test_df.shape: ' + str(test_df.shape))
    print('train_label.shape: ' + str(train_label.shape))
    print('test_label.shape: ' + str(test_label.shape))

    r"""do preprocess for test and train sample"""

    # with tf.device('/cpu:0'):
    #     bert_preprocess_model = make_bert_preprocess_model(['text_pre', 'bio_info'])
    # test_text = [np.array([train_df[0]]),
    #              np.array([train_df[0]])]
    # print('test_text: ' + str(test_text))
    #
    # with tf.device('/cpu:0'):
    #     text_preprocessed = bert_preprocess_model(test_text)
    #
    # keys = list(text_preprocessed.keys())
    # input_mask_key = keys[0]
    # input_type_ids_key = keys[1]
    # input_word_ids_key = keys[2]
    #
    # input_mask = text_preprocessed[input_mask_key]
    # input_type_ids = text_preprocessed[input_type_ids_key]
    # input_word_ids = text_preprocessed[input_word_ids_key]
    #
    # tf.keras.utils.plot_model(bert_preprocess_model, to_file='bert_preprocess.png', show_dtype=True)
    #
    # print(f'Keys          : {list(keys)}')
    # print(f'Shape Word Ids: {input_word_ids.shape}')
    # print_indented_key_value(key='Word Ids      : ', value=input_word_ids, intend_num=3)
    # print(f'Shape Mask    : {input_mask.shape}')
    # print(f'Input Mask    : {input_mask}')
    # print(f'Shape Type Ids: {input_type_ids.shape}')
    # print(f'Type Ids      : {input_type_ids}')

    r"""do preprocess for test and train"""

    # test_classifier_model = build_classifier_model(2)
    # bert_raw_result = test_classifier_model(text_preprocessed)
    # print(tf.sigmoid(bert_raw_result))

    r"""do preprocess for test and train of GLUE"""

    tfds_name = 'glue/cola'

    tfds_info = tfds.builder(tfds_name).info

    sentence_features = list(tfds_info.features.keys())
    sentence_features.remove('idx')
    sentence_features.remove('label')

    available_splits = list(tfds_info.splits.keys())
    train_split = 'train'
    validation_split = 'validation'
    test_split = 'test'
    if tfds_name == 'glue/mnli':
        validation_split = 'validation_matched'
        test_split = 'test_matched'

    num_classes = tfds_info.features['label'].num_classes
    num_examples = tfds_info.splits.total_num_examples

    print(f'Using {tfds_name} from TFDS')
    print(f'This dataset has {num_examples} examples')
    print(f'Number of classes: {num_classes}')
    print(f'Features {sentence_features}')
    print(f'Splits {available_splits}')

    with tf.device('/job:localhost'):
        # batch_size=-1 is a way to load the dataset into memory
        in_memory_ds = tfds.load(tfds_name, batch_size=-1, shuffle_files=True)
    print(in_memory_ds)
    # The code below is just to show some samples from the selected dataset
    print(f'Here are some sample rows from {tfds_name} dataset')
    sample_dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds[train_split])

    labels_names = tfds_info.features['label'].names
    print(labels_names)
    print()

    sample_i = 1
    for sample_row in sample_dataset.take(5):
        samples = [sample_row[feature] for feature in sentence_features]
        print(f'sample row {sample_i}')
        for sample in samples:
            print(sample.numpy())
        sample_label = sample_row['label']

        print(f'label: {sample_label} ({labels_names[sample_label]})')
        print()
        sample_i += 1
    return

    # # tokenizer = bert_preprocess.tokenize
    # # tok = list([__bert_tokenize(item, tokenizer) for item in text_pre_df])
    # # print(tok)
    # # tok = bert_preprocess.tokenize(tf.constant(text_pre_df.tolist()))
    # tok = bert_preprocess.tokenize(tf.constant(['Hello baby', 'Hello me']))
    # # print(tok)
    # with tf.device('/cpu:0'):
    #     seq_length = 128  # Your choice here.
    #     bert_pack_inputs = hub.KerasLayer(
    #         bert_preprocess.bert_pack_inputs,
    #         arguments=dict(seq_length=seq_length))  # Optional argument.
    #     encoder_inputs = bert_pack_inputs(tok)
    #     print(encoder_inputs)
    #     #     print(type(tok))
    # text_preprocessed = bert_preprocess.bert_pack_inputs(tok, tf.constant(128))
    #
    # make_bert_preprocess_model(sentence_features=text_pre_df, seq_length=128)
    # return
    # print('Shape Word Ids : ', text_preprocessed['input_word_ids'].shape)
    # print('Word Ids       : ', text_preprocessed['input_word_ids'])
    # print('Shape Mask     : ', text_preprocessed['input_mask'].shape)
    # print('Input Mask     : ', text_preprocessed['input_mask'][0, :16])
    # print('Shape Type Ids : ', text_preprocessed['input_type_ids'].shape)
    # print('Type Ids       : ', text_preprocessed['input_type_ids'][0, :16])


AUTOTUNE = tf.data.AUTOTUNE


def test_val_train_split(df):
    xdf = df['text_pre']
    ydf = df['is_rumour']
    r"""dividing the dataset into train, test, cv with 0.6, 0.2, 0.2"""
    x, x_test, y, y_test = train_test_split(xdf, ydf, test_size=0.2, train_size=0.8)
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.25, train_size=0.75)
    return numpy.array(x_train), numpy.array(x_test), numpy.array(x_cv), numpy.array(y_train), numpy.array(
        y_test), numpy.array(y_cv)


def load_dataset_from_phemeds(df, batch_size,
                              bert_preprocess_model):
    # cols = df.select_dtypes(include=['object'])
    # for col in cols.columns.values:
    #     df[col] = df[col].fillna('')
    df_features = df.copy()
    titanic_labels = df_features['is_rumour']
    inputs = {}

    for name, column in df_features.items():
        dtype = column.dtype
        if name != 'text' and name != 'is_rumour':
            continue
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    numeric_inputs = {name: input for name, input in inputs.items()
                      if input.dtype == tf.float32}

    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = layers.Normalization()
    norm.adapt(np.array(df[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)

    preprocessed_inputs = [all_numeric_inputs]
    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue
        lookup = tf.keras.layers.StringLookup(vocabulary=np.unique(df_features[name]))
        one_hot = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)
    print('inp: ' + str(inputs))
    print('inp: ' + str(inputs['text']))
    print('all_numeric_inputs: ' + str(all_numeric_inputs))
    print('preprocessed_inputs: ' + str(preprocessed_inputs))

    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

    titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

    tf.keras.utils.plot_model(model=titanic_preprocessing, rankdir="LR", dpi=72, show_shapes=True)
    titanic_features_dict = {}
    for name, value in df_features.items():

        if name != 'text' and name != 'is_rumour':
            continue
        titanic_features_dict[name] = np.array(value)

    print(titanic_features_dict)
    print('features_dict')
    features_dict = {name: values for name, values in titanic_features_dict.items()}
    print(len(features_dict['text']))

    # s = titanic_preprocessing(features_dict)
    # print('s')
    # print(s)

    titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict['text'], titanic_labels))
    titanic_ds = titanic_ds.shuffle(df.shape[0])
    titanic_ds = titanic_ds.repeat()
    titanic_ds = titanic_ds.batch(batch_size)
    print(titanic_ds)

    features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)
    print(features_ds)
    return
    ds = tf.data.Dataset.from_tensor_slices(s)
    print(ds.take(3))

    dd = ds.map(lambda ex: (bert_preprocess_model(ex), ex['text']))
    print(dd)

    return
    x_train, x_test, x_cv, y_train, y_test, y_cv = test_val_train_split(df)

    train_example_num = len(y_train)
    test_example_num = len(y_test)
    validation_example_num = len(y_cv)
    train_ds = DataFrame()
    train_ds['label'] = y_train
    train_ds['sentence'] = x_train

    print(train_ds.head())
    print(type(train_ds))
    # x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    print(y_train)
    print(type(x_train))
    print(type(y_train))
    print((y_train))
    ds = tf.data.Dataset.from_tensors(train_ds)
    # print(ds)
    # TensorSliceDataset
    # element_spec = {'label': TensorSpec(shape=(), dtype=tf.int32, name=None), ...}

    ds.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
    # dataset = tf.data.Dataset.from_tensor_slices(df[split])
    # num_examples = info.splits[split].num_examples

    # if is_training:
    #     dataset = dataset.shuffle(num_examples)
    #     dataset = dataset.repeat()
    # dataset = dataset.batch(batch_size)
    print('train_dss')
    print(train_ds)
    # dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
    print('datassettt')
    # print(d)
    # dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    # return dataset, num_examples


def load_dataset_from_tfds(df, info, split, batch_size,
                           bert_preprocess_model):
    is_training = split.startswith('train')
    dataset = tf.data.Dataset.from_tensor_slices(df[split])
    num_examples = info.splits[split].num_examples

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    print('datassettt1')
    print(dataset)
    dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
    print('datassettt')
    print(dataset)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return dataset, num_examples


def build_classifier_model(num_classes):
    class Classifier(tf.keras.Model):
        def __init__(self, num_classes):
            super(Classifier, self).__init__(name="prediction")
            self.encoder = hub.KerasLayer(bert_model_name, trainable=True)
            self.dropout = tf.keras.layers.Dropout(0.1)
            self.dense = tf.keras.layers.Dense(num_classes)

        def call(self, preprocessed_text):
            encoder_outputs = self.encoder(preprocessed_text)
            pooled_output = encoder_outputs["pooled_output"]
            x = self.dropout(pooled_output)
            x = self.dense(x)
            return x

    model = Classifier(num_classes)
    return model


def get_configuration(glue_task):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if glue_task == 'glue/cola':
        metrics = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
    else:
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)

    return metrics, loss


def make_bert_preprocess_model(sentence_features, seq_length=128):
    """Returns Model mapping string features to BERT inputs.

        Args:
          sentence_features: a list with the names of string-valued features.
          seq_length: an integer that defines the sequence length of BERT inputs.

        Returns:
          A Keras Model that can be called on a list or dict of string Tensors
          (with the order or names, resp., given by sentence_features) and
          returns a dict of tensors for input to BERT.

        1- Tokenize the text to word pieces.
        2- Pack inputs. The details (start/end token ids, dict of output tensors)
        are model-dependent, so this gets loaded from the SavedModel.
        """

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features]

    bert_preprocess = hub.load(bert_preprocess_model_name)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]

    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=seq_length),
                            name='packer')
    model_inputs = packer(segments)
    return tf.keras.Model(input_segments, model_inputs)
