# # https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
#
# # BERT imports
# import torch
# from keras.utils import pad_sequences
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# # from keras.preprocessing import pad_sequences
# from sklearn.model_selection import train_test_split
# from pytorch_pretrained_bert import BertTokenizer, BertConfig
# from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
# from tqdm import tqdm, trange
# import pandas as pd
# import io
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Upload the train file from your local drive
# # from google.colab import files
# #
# # uploaded = files.upload()
# #
# # # Upload data from Google Drive
# # from google.colab import drive
# #
# # drive.mount('/drive')
# #
# # #
# # # queries are stored in the variable query_data_train
# # # correct intent labels are stored in the variable labels
# # #
# #
# # # add special tokens for BERT to work properly
# # sentences = ["[CLS] " + query + " [SEP]" for query in query_data_train]
# # print(sentence[0])
#
# # Tokenize with BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
# print("Tokenize the first sentence:")
# print(tokenized_texts[0])
#
# # Set the maximum sequence length.
# MAX_LEN = 128
# # Pad our input tokens
# input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
#                           maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
# input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
#
# # Create attention masks
# attention_masks = []
# # Create a mask of 1s for each token followed by 0s for padding
# for seq in input_ids:
#     seq_mask = [float(i > 0) for i in seq]
#     attention_masks.append(seq_mask)
#
# # Use train_test_split to split our data into train and validation sets for training
# train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
#                                                                                     random_state=2018, test_size=0.1)
# train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
#                                                        random_state=2018, test_size=0.1)
#
# # Convert all of our data into torch tensors, the required datatype for our model
# train_inputs = torch.tensor(train_inputs)
# validation_inputs = torch.tensor(validation_inputs)
# train_labels = torch.tensor(train_labels)
# validation_labels = torch.tensor(validation_labels)
# train_masks = torch.tensor(train_masks)
# validation_masks = torch.tensor(validation_masks)
#
# # Select a batch size for training.
# batch_size = 32
#
# # Create an iterator of our data with torch DataLoader
# train_data = TensorDataset(train_inputs, train_masks, train_labels)
# train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
# validation_sampler = SequentialSampler(validation_data)
# validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
#
# # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
# nb_labels = 2
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=nb_labels)
# model.cuda()
#
# # # BERT model summary
# # BertForSequenceClassification(
# #   (bert): BertModel(
# #     (embeddings): BertEmbeddings(
# #       (word_embeddings): Embedding(30522, 768, padding_idx=0)
# #       (position_embeddings): Embedding(512, 768)
# #       (token_type_embeddings): Embedding(2, 768)
# #       (LayerNorm): BertLayerNorm()
# #       (dropout): Dropout(p=0.1)
# #     )
# #     (encoder): BertEncoder(
# #       (layer): ModuleList(
# #         (0): BertLayer(
# #           (attention): BertAttention(
# #             (self): BertSelfAttention(
# #               (query): Linear(in_features=768, out_features=768, bias=True)
# #               (key): Linear(in_features=768, out_features=768, bias=True)
# #               (value): Linear(in_features=768, out_features=768, bias=True)
# #               (dropout): Dropout(p=0.1)
# #             )
# #             (output): BertSelfOutput(
# #               (dense): Linear(in_features=768, out_features=768, bias=True)
# #               (LayerNorm): BertLayerNorm()
# #               (dropout): Dropout(p=0.1)
# #             )
# #           )
# #           (intermediate): BertIntermediate(
# #             (dense): Linear(in_features=768, out_features=3072, bias=True)
# #           )
# #           (output): BertOutput(
# #             (dense): Linear(in_features=3072, out_features=768, bias=True)
# #             (LayerNorm): BertLayerNorm()
# #             (dropout): Dropout(p=0.1)
# #           )
# #         )
# #         '
# #         '
# #         '
# #       )
# #     )
# #     (pooler): BertPooler(
# #       (dense): Linear(in_features=768, out_features=768, bias=True)
# #       (activation): Tanh()
# #     )
# #   )
# #   (dropout): Dropout(p=0.1)
# #   (classifier): Linear(in_features=768, out_features=2, bias=True)
# # )
#
# # BERT fine-tuning parameters
# param_optimizer = list(model.named_parameters())
# no_decay = ['bias', 'gamma', 'beta']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#      'weight_decay_rate': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#      'weight_decay_rate': 0.0}
# ]
#
# optimizer = BertAdam(optimizer_grouped_parameters,
#                      lr=2e-5,
#                      warmup=.1)
#
#
# # Function to calculate the accuracy of our predictions vs labels
# def flat_accuracy(preds, labels):
#     pred_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return np.sum(pred_flat == labels_flat) / len(labels_flat)
#
#
# # Store our loss and accuracy for plotting
# train_loss_set = []
# # Number of training epochs
# epochs = 4
#
# # BERT training loop
# for _ in trange(epochs, desc="Epoch"):
#
#     ## TRAINING
#
#     # Set our model to training mode
#     model.train()
#     # Tracking variables
#     tr_loss = 0
#     nb_tr_examples, nb_tr_steps = 0, 0
#     # Train the data for one epoch
#     for step, batch in enumerate(train_dataloader):
#         # Add batch to GPU
#         batch = tuple(t.to(device) for t in batch)
#         # Unpack the inputs from our dataloader
#         b_input_ids, b_input_mask, b_labels = batch
#         # Clear out the gradients (by default they accumulate)
#         optimizer.zero_grad()
#         # Forward pass
#         loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
#         train_loss_set.append(loss.item())
#         # Backward pass
#         loss.backward()
#         # Update parameters and take a step using the computed gradient
#         optimizer.step()
#         # Update tracking variables
#         tr_loss += loss.item()
#         nb_tr_examples += b_input_ids.size(0)
#         nb_tr_steps += 1
#     print("Train loss: {}".format(tr_loss / nb_tr_steps))
#
#     ## VALIDATION
#
#     # Put model in evaluation mode
#     model.eval()
#     # Tracking variables
#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_steps, nb_eval_examples = 0, 0
#     # Evaluate data for one epoch
#     for batch in validation_dataloader:
#         # Add batch to GPU
#         batch = tuple(t.to(device) for t in batch)
#         # Unpack the inputs from our dataloader
#         b_input_ids, b_input_mask, b_labels = batch
#         # Telling the model not to compute or store gradients, saving memory and speeding up validation
#         with torch.no_grad():
#             # Forward pass, calculate logit predictions
#             logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#             # Move logits and labels to CPU
#         logits = logits.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()
#         tmp_eval_accuracy = flat_accuracy(logits, label_ids)
#         eval_accuracy += tmp_eval_accuracy
#         nb_eval_steps += 1
#     print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
#
# # plot training performance
# plt.figure(figsize=(15, 8))
# plt.title("Training loss")
# plt.xlabel("Batch")
# plt.ylabel("Loss")
# plt.plot(train_loss_set)
# plt.show()
#
# # load test data
# sentences = ["[CLS] " + query + " [SEP]" for query in query_data_test]
# labels = intent_data_label_test
#
# # tokenize test data
# tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
# MAX_LEN = 128
# # Pad our input tokens
# input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
#                           maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
# input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# # Create attention masks
# attention_masks = []
# # Create a mask of 1s for each token followed by 0s for padding
# for seq in input_ids:
#     seq_mask = [float(i > 0) for i in seq]
#     attention_masks.append(seq_mask)
#
# # create test tensors
# prediction_inputs = torch.tensor(input_ids)
# prediction_masks = torch.tensor(attention_masks)
# prediction_labels = torch.tensor(labels)
# batch_size = 32
# prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
# prediction_sampler = SequentialSampler(prediction_data)
# prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
#
# ## Prediction on test set
# # Put model in evaluation mode
# model.eval()
# # Tracking variables
# predictions, true_labels = [], []
# # Predict
# for batch in prediction_dataloader:
#     # Add batch to GPU
#     batch = tuple(t.to(device) for t in batch)
#     # Unpack the inputs from our dataloader
#     b_input_ids, b_input_mask, b_labels = batch
#     # Telling the model not to compute or store gradients, saving memory and speeding up prediction
#     with torch.no_grad():
#         # Forward pass, calculate logit predictions
#         logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#     # Move logits and labels to CPU
#     logits = logits.detach().cpu().numpy()
#     label_ids = b_labels.to('cpu').numpy()
#     # Store predictions and true labels
#     predictions.append(logits)
#     true_labels.append(label_ids)
#
# # Import and evaluate each test batch using Matthew's correlation coefficient
# from sklearn.metrics import matthews_corrcoef
#
# matthews_set = []
# for i in range(len(true_labels)):
#     matthews = matthews_corrcoef(true_labels[i],
#                                  np.argmax(predictions[i], axis=1).flatten())
#     matthews_set.append(matthews)
#
# # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
# flat_predictions = [item for sublist in predictions for item in sublist]
# flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
# flat_true_labels = [item for sublist in true_labels for item in sublist]
#
# print('Classification accuracy using BERT Fine Tuning: {0:0.2%}'.format(
#     matthews_corrcoef(flat_true_labels, flat_predictions)))
