
import os
from lib.training_modules.bert.all_features_bert.preprocess.bert_preprocessing import \
    BertPreprocessing
from lib.training_modules.bert.all_features_bert.train.bert_train import BertTrain
from lib.dataset_repositories.pheme.read_pheme import read_pheme
from lib.training_modules.bert.analysis.bert_model_analysis import BertModelAnalysis
from lib.training_modules.bert.bert_configurations import ENABLE_GPU
r"""
    1- Read dataset...
    2- Do preprocess on it
    3- Run BiLSTM
    4- Run Bert
"""
# BertModelAnalysis("", '').plot_bert_evaluation_metrics('','','','')
if not ENABLE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


train_df, val_df, test_df = read_pheme()
# train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, label_classes, train_len, validation_len, test_len, \
# bert_preprocess_model = BertPreprocessingImpl().start(train_df, val_df, test_df)
#
# BertModelImpl(train_tensor_dataset=train_tensor_dataset, val_tensor_dataset=val_tensor_dataset,
#               test_tensor_dataset=test_tensor_dataset, num_classes=label_classes, train_len=train_len,
#               validation_len=validation_len, test_len=test_len, bert_preprocess_model=bert_preprocess_model).start()
bert_preprocess = BertPreprocessing(train_df, val_df, test_df)
encoded_dataset, tokenizer = bert_preprocess.start()

BertTrain(encoded_dataset, tokenizer).start()
# BertNew(train_df, val_df, test_df).start()
# train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, train_len, validation_len, test_len, bi_lstm_preprocess_model = \
#     BiLstmPreprocess().start(
#         train_df, val_df, test_df)
# BiLstmModelImpl(train_tensor_dataset, val_tensor_dataset, test_tensor_dataset, train_len, validation_len, test_len,
#                 bi_lstm_preprocess_model, train_df, val_df, test_df).start()

#
# def __get_x_y_from_df( df):
#     y = df['text']
#     x = df['is_rumour']
#     return x, y
#
#
# x_train, y_train = __get_x_y_from_df(train_df)
# x_test, y_test = __get_x_y_from_df(test_df)
# x_val, y_val = __get_x_y_from_df(val_df)
#
# do_bi_lstm(x_train, x_test, y_train, y_test)
# BertModelReload(train_tensor_dataset)
# bert_test(preprocessed_df)
#
# print('preprocessed ds')
# print(preprocessed_df.shape)

# x_train, x_test, y_train, y_test = bi_lstm.data_reshape(df=dataframe)
# bi_lstm.run_bi_lstm(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, max_len=64, epoch=1)

# do_bert(preprocessed_df)


# do_bi_lstm(preprocessed_df)
#
# do_bert(preprocessed_df)

# import tensorflow as tf
# import tensorflow_hub as hub
#
#
# def build_classifier_model():
#     bert_model_name = get_bert_model_name()
#     bert_preprocess_model_name = get_bert_preprocess_model_name()
#
#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
#     print('text input')
#     print(text_input)
#
#     preprocessing_layer = hub.KerasLayer(bert_preprocess_model_name, name='preprocessing')
#     encoder_inputs = preprocessing_layer(text_input)
#     encoder = hub.KerasLayer(bert_model_name, trainable=True, name='BERT_encoder')
#     outputs = encoder(encoder_inputs)
#
#     net = outputs['pooled_output']
#     net = tf.keras.layers.Dropout(0.1)(net)
#     net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
#
#     return tf.keras.Model(text_input, net)
#
#
# sample_text = ['this is such an amazing movie!', 'baby i know', 'my love is you']
#
# classifier_model = build_classifier_model()
# print(classifier_model.summary())
# print(tf.constant(sample_text))
# with tf.device('/cpu:0'):
#     bert_raw_result = classifier_model(tf.constant(sample_text))
#     print(tf.sigmoid(bert_raw_result))
#     tf.keras.utils.plot_model(classifier_model)
#     loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     metrics = tf.metrics.BinaryAccuracy()
#     epochs = 5
#     steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
#     num_train_steps = steps_per_epoch * epochs
#     num_warmup_steps = int(0.1 * num_train_steps)
#
#     init_lr = 3e-5
#     optimizer = optimization.create_optimizer(init_lr=init_lr,
#                                               num_train_steps=num_train_steps,
#                                               num_warmup_steps=num_warmup_steps,
#                                               optimizer_type='adamw')
#
#     classifier_model.compile(optimizer=optimizer,
#                              loss=loss,
#                              metrics=metrics)

# import torch
# from transformers import AutoTokenizer, AutoModel
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

# bert_preprocess = KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
# preprocessor = hub.KerasLayer(
#     "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# encoder_inputs = preprocessor(text_input)
# import torch

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# example_text = '[CLS] ' + "The man I've seen." + ' [SEP] ' + 'He bough a gallon of milk.' + ' [SEP] '
# print('sample text is : ' + example_text)
# tokenized_text = tokenizer.tokenize(text=example_text)
#
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokens=tokenized_text)
#
# for tup in zip(tokenized_text, indexed_tokens):
#     print('{:<12}{:>6,}'.format(tup[0], tup[1]))
#
# segments_ids = [1] * len(tokenized_text)
#
# print('segments_ids are ' + str(segments_ids))
# tokens_tensor = torch.tensor([indexed_tokens])
# print(tokens_tensor)
# print('tokens_tensor are ' + str(tokens_tensor))
#
# segments_tensors = torch.tensor([segments_ids])
# print(segments_tensors)
# print('segments_tensors are ' + str(segments_tensors))
#
# model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
#
#
# model.eval()
# with torch.no_grad():
#     outputs = model(tokens_tensor, segments_tensors)
#     hidden_states = outputs[2]
#     print(outputs[0])
#     print(outputs[1])
#     print(outputs[2])
#
#     print(hidden_states)
