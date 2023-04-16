
from matplotlib import pyplot as plt

from lib.read_datasets.pheme.read_pheme_ds import read_pheme_ds
from lib.training_modules.bert.bert_model_name import get_bert_model_name, get_bert_preprocess_model_name
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
from pandas import DataFrame


read_pheme_ds()
def bert_preprocess(new_text):
    bert_preprocess_model_name = get_bert_preprocess_model_name()

    print('ssss')
    bert_preprocess_model = hub.KerasLayer(bert_preprocess_model_name, name='preprocessing')
    print(new_text)

    with tf.device('/cpu:0'):
        text_preprocessed = bert_preprocess_model(new_text)
    print('sssdddsdddd')

    keys = list(text_preprocessed.keys())
    input_mask_key = keys[0]
    input_type_ids_key = keys[1]
    input_word_ids_key = keys[2]

    input_mask = text_preprocessed[input_mask_key]
    input_type_ids = text_preprocessed[input_type_ids_key]
    input_word_ids = text_preprocessed[input_word_ids_key]

    print(f'Keys       : {list(text_preprocessed.keys())}')
    print(f'Shape      : {input_word_ids.shape}')
    print(f'Word Ids   : {input_word_ids}')
    print(f'Input Mask : {input_mask}')
    print(f'Type Ids   : {input_type_ids}')


def bert_test(preprocessed_df):
    # url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    #
    # dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
    #                                   untar=True, cache_dir='.',
    #                                   cache_subdir='')
    #
    # dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    #
    # train_dir = os.path.join(dataset_dir, 'train')
    #
    # # remove unused folders to make it easier to load the data
    # remove_dir = os.path.join(train_dir, 'unsup')
    # shutil.rmtree(remove_dir)
    #
    # AUTOTUNE = tf.data.AUTOTUNE
    # batch_size = 32
    # seed = 42

    bert_model_name = get_bert_model_name()
    bert_preprocess_model_name = get_bert_preprocess_model_name()

    # print(preprocessed_df['text'].tolist())
    print(len(preprocessed_df['text'].tolist()))
    df = DataFrame()
    df['text_pre'] = preprocessed_df['text_pre']
    df['user.description_pre'] = preprocessed_df['user.description_pre']

    # with tf.device('/cpu:0'):
    df['text_bert_pre'] = df['text'].apply(lambda x: bert_preprocess(x))
    # raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    #     'aclImdb/train',
    #     batch_size=batch_size,
    #     validation_split=0.2,
    #     subset='training',
    #     seed=seed)
    print(len(preprocessed_df['text_bert_pre'].tolist()))
    print(len(preprocessed_df['text_bert_pre'].head()))
    return
    class_names = ['rumour', 'non rumour']
    # train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    #
    # val_ds = tf.keras.utils.text_dataset_from_directory(
    #     'aclImdb/train',
    #     batch_size=batch_size,
    #     validation_split=0.2,
    #     subset='validation',
    #     seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test',
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # print(train_ds.take(1))
    # print(train_ds.take(2))
    # print(len(train_ds))
    # print(len(test_ds))
    # for text_batch, label_batch in train_ds.take(1):
    #     print(label_batch.numpy)
    #     print(text_batch)
    #     for i in range(3):
    #         print(f'Review: {text_batch.numpy()[i]}')
    #         label = label_batch.numpy()[i]
    #         print(f'Label : {label} ({class_names[label]})')

    text_test = ['this is such an amazing movie!', 'my love ', 'hateful man']

    with tf.device('/cpu:0'):
        bert_preprocess_model = hub.KerasLayer(bert_preprocess_model_name, name='preprocessing')

        text_preprocessed = bert_preprocess_model(text_test)

        keys = list(text_preprocessed.keys())
        input_mask_key = keys[0]
        input_type_ids_key = keys[1]
        input_word_ids_key = keys[2]

        input_mask = text_preprocessed[input_mask_key]
        input_type_ids = text_preprocessed[input_type_ids_key]
        input_word_ids = text_preprocessed[input_word_ids_key]

        print(f'Keys       : {list(text_preprocessed.keys())}')
        print(f'Shape      : {input_word_ids.shape}')
        print(f'Word Ids   : {input_word_ids}')
        print(f'Input Mask : {input_mask}')
        print(f'Type Ids   : {input_type_ids}')

        classifier_model = build_classifier_model()
        bert_raw_result = classifier_model(tf.constant(text_test))
        print(tf.sigmoid(bert_raw_result))

        tf.keras.utils.plot_model(classifier_model)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()

        epochs = 5
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1 * num_train_steps)

        from bert import optimization

        init_lr = 3e-5
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  # optimizer_type='adamw'
                                                  )

        classifier_model.compile(optimizer=optimizer,
                                 loss=loss,
                                 metrics=metrics, )
        print(f'Training model with {bert_model_name}')
        history = classifier_model.fit(x=train_ds,
                                       validation_data=val_ds,
                                       epochs=epochs)

        loss, accuracy = classifier_model.evaluate(test_ds)

        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')

        history_dict = history.history
        print(history_dict.keys())

        acc = history_dict['binary_accuracy']
        val_acc = history_dict['val_binary_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        # r is for "solid red line"
        plt.plot(epochs, loss, 'r', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        dataset_name = 'imdb'
        saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

        classifier_model.save(saved_model_path, include_optimizer=False)

        reloaded_model = tf.saved_model.load(saved_model_path)

        def print_my_examples(inputs, results):
            result_for_printing = \
                [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                 for i in range(len(inputs))]
            print(*result_for_printing, sep='\n')
            print()

        examples = [
            'this is such an amazing movie!',  # this is the same sentence tried earlier
            'The movie was great!',
            'The movie was meh.',
            'The movie was okish.',
            'The movie was terrible...'
        ]

        reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
        original_results = tf.sigmoid(classifier_model(tf.constant(examples)))

        print('Results from the saved model:')
        print_my_examples(examples, reloaded_results)
        print('Results from the model in memory:')
        print_my_examples(examples, original_results)


def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    # preprocessing_layer = hub.KerasLayer(bert_preprocess_model_name, name='preprocessing')
    # encoder_inputs = preprocessing_layer(text_input)
    # encoder = hub.KerasLayer(bert_model_name, trainable=True, name='BERT_encoder')
    # outputs = encoder(encoder_inputs)
    # net = outputs['pooled_output']
    # net = tf.keras.layers.Dropout(0.1)(net)
    # net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    # return tf.keras.Model(text_input, net)


