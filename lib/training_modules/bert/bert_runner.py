import os
import tensorflow_datasets as tfds
# from official.nlp import optimization
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # A dependency of the preprocessing model
import tensorflow_addons as tfa
import transformers
# from bert import optimization
# from torch.backends.opt_einsum import strategy
# from lib.training_modules.bert import optimization
from sklearn.model_selection import train_test_split

from lib.test3 import bert_model_name, build_classifier_model
from lib.training_modules.bert.bert_model_name import get_bert_model_name, get_bert_preprocess_model_name
from lib.utils.log.logger import print_indented, print_indented_key_value

AUTOTUNE = tf.data.AUTOTUNE


def run_bert_process(df, train_dataset, train_data_size, num_classes):
    epochs = 1
    batch_size = 32
    init_lr = 2e-5

    bert_model_name = get_bert_model_name()
    bert_preprocess_model_name = get_bert_preprocess_model_name()

    # sentence_features, available_splits, num_examples, num_classes = \
    #     __ds_statistics(df)

    # bert_preprocess_model = make_bert_preprocess_model(sentence_features,
    #                                                    bert_preprocess_model_name)

    metrics, loss = get_configuration('')

    print(f'metrics: {metrics}, loss: {loss}')

    # train_dataset, train_data_size = load_dataset_from_tfds(
    #     df=df,
    #     dicted_ds=dicted_ds,
    #     num_examples=num_examples,
    #     batch_size=batch_size,
    #     bert_preprocess_model=bert_preprocess_model,
    #     is_training=True)
    steps_per_epoch = train_data_size // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = num_train_steps // 10
    print(f'Fine tuning {bert_model_name} model')
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

    print('before fit')

    for item in train_dataset.take(2):
        print('item')
        print(item)
    print(train_dataset)
    print('before fit')
    classifier_model.fit(
        x=train_dataset,
        # validation_data=train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        # validation_steps=validation_steps,
    )
    main_save_path = './my_models'
    bert_type = bert_model_name.split('/')[-2]
    saved_model_name = f'PHEMEmODeL_{bert_type}'

#
# def build_classifier_model(num_classes):
#     class Classifier(tf.keras.Model):
#         def __init__(self, num_classes):
#             super(Classifier, self).__init__(name="prediction")
#             self.encoder = hub.KerasLayer(bert_model_name, trainable=True)
#             self.dropout = tf.keras.layers.Dropout(0.1)
#             self.dense = tf.keras.layers.Dense(num_classes)
#
#         def call(self, preprocessed_text):
#             encoder_outputs = self.encoder(preprocessed_text)
#             pooled_output = encoder_outputs["pooled_output"]
#             x = self.dropout(pooled_output)
#             x = self.dense(x)
#             return x
#
#     model = Classifier(num_classes)
#     return model


def make_bert_preprocess_model(sentence_features,
                               bert_preprocess_model_name,
                               seq_length=32):
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
                            # arguments=dict(seq_length=seq_length),
                            name='packer')
    model_inputs = packer(segments)
    return tf.keras.Model(input_segments, model_inputs)


def load_dataset_from_tfds(df, dicted_ds, num_examples, batch_size,
                           bert_preprocess_model, is_training):
    # is_training = split.startswith('train')
    # dataset = tf.data.Dataset.from_tensor_slices(df[split])
    print(dicted_ds.shape)
    dataset = tf.data.Dataset.from_tensor_slices((dicted_ds, df['is_rumour']))
    print(f'dataset after slice {dataset}')

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    for text, label in dataset.take(2):
        print(text)
        print(label)
        v = bert_preprocess_model(text)
        print(v)
    dataset = dataset.apply(lambda ex: bert_preprocess_model(ex))
    print('datassettt')
    print(dataset)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return dataset, num_examples


def get_configuration(glue_task):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if glue_task == 'glue/cola':
        metrics = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
    else:
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)

    return metrics, loss


def __ds_statistics(df):
    tfds_name = 'PHEME'

    sentence_features = list(['text'])

    available_splits = list(['train', 'validation', 'test'])
    num_classes = len(df['is_rumour'].value_counts())
    num_examples = df.shape[0]

    print(f'Using {tfds_name} from TFDS')
    print(f'This dataset has {num_examples} examples')
    print(f'Number of classes: {num_classes}')
    print(f'Features {sentence_features}')
    print(f'Splits {available_splits}')

    return sentence_features, available_splits, num_examples, num_classes
