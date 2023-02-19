# https://medium.com/analytics-vidhya/text-classification-from-bag-of-words-to-bert-part-6-bert-2c3a5821ed16

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup
import re

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from torch.nn import BCEWithLogitsLoss, Sigmoid

from tqdm.notebook import tqdm, trange

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

#
from lib.read_datasets.pheme.read_pheme_ds import read_pheme_ds
from lib.training_modules.bert.preprocess.bert_preprocessing_impl import BertPreprocessingImpl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strip(text):
    soup = BeautifulSoup(text, 'html.parser')
    text = re.sub('\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text


df_train = read_pheme_ds()
y = df_train['is_rumour']
x = df_train['text']
x_train, x_val, x_test, y_train, y_val, y_test = \
    BertPreprocessingImpl.train_val_test_split(x, y, 0.7, 0.1, 0.2)

df_train["text"] = df_train["text"].apply(strip)
df_test["text"] = df_test["text"].apply(strip)

train_sentences = df_train["text"]
test_sentences = df_test["text"]
train_sentences = ["[CLS] "+ i + " [SEP]"for i in train_sentences]
test_sentences = ["[CLS] "+ i + " [SEP]"for i in test_sentences]
train_sentences[0], test_sentences[0]


