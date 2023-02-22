# import random
# import copy
# import time
# import pandas as pd
# import numpy as np
# import gc
# import re
# import torch
# from torchtext import data
# # import spacy
# from tqdm import tqdm_notebook, tnrange
# from tqdm.auto import tqdm
#
# from lib.training_modules.bilstm.bilstm import BiLstm
#
# from collections import Counter
# from textblob import TextBlob
# from nltk import word_tokenize
#
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.autograd import Variable
# # from torchtext.data import Example
# from sklearn.metrics import f1_score
# # import torchtext
# import os
#
# from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# # cross validation and metrics
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import f1_score
# from torch.optim.optimizer import Optimizer
# from unidecode import unidecode
#
# from sklearn.preprocessing import StandardScaler
# from textblob import TextBlob
# from multiprocessing import Pool
# from functools import partial
# import numpy as np
# from sklearn.decomposition import PCA
# import torch as t
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class BiLstmImpl2(BiLstm):
#     r"""
#         embed_size: how big is each word vector
#         max_features: how many unique words to use (i.e num rows in embedding vector)
#         max_len: max number of words in a question to use
#         batch_size: how many samples to process at once
#         n_epochs: how many times to iterate over all samples
#         n_splits: Number of K-fold Splits
#         resource => https://www.kaggle.com/code/mlwhiz/bilstm-pytorch-and-keras
#     """
#
#     def __init__(self, embed_size=300, max_features=120000, max_len=70, batch_size=512, n_epochs=5, n_splits=5, seed=10,
#                  debug=0):
#         tqdm.pandas(desc='Progress')
#         self.embed_size = embed_size
#         self.max_features = max_features
#         self.max_len = max_len
#         self.batch_size = batch_size
#         self.n_epochs = n_epochs
#         self.n_splits = n_splits
#         self.SEED = seed
#         self.debug = debug
#         self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
#
#         # self.lstm_model = None
#         # self.bi_lstm_model = None
#         # self.results = DataFrame()
#
#     def __seed_everything(self):
#         random.seed(self.SEED)
#         os.environ['PYTHONHASHSEED'] = str(self.SEED)
#         np.random.seed(self.SEED)
#         torch.manual_seed(self.SEED)
#         torch.cuda.manual_seed(self.SEED)
#         torch.backends.cudnn.deterministic = True
#
#     def load_glove(self, word_index):
#         EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
#
#         def get_coefs(word, *arr):
#             return word, np.asarray(arr, dtype='float32')[:300]
#
#         embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
#
#         all_embs = np.stack(embeddings_index.values())
#         emb_mean, emb_std = -0.005838499, 0.48782197
#         embed_size = all_embs.shape[1]
#
#         # word_index = tokenizer.word_index
#         nb_words = min(self.max_features, len(word_index))
#         embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#         for word, i in word_index.items():
#             if i >= self.max_features: continue
#             embedding_vector = embeddings_index.get(word)
#             # ALLmight
#             if embedding_vector is not None:
#                 embedding_matrix[i] = embedding_vector
#             else:
#                 embedding_vector = embeddings_index.get(word.capitalize())
#                 if embedding_vector is not None:
#                     embedding_matrix[i] = embedding_vector
#         return embedding_matrix
#
#     def load_fasttext(self, word_index):
#         EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
#
#         def get_coefs(word, *arr):
#             return word, np.asarray(arr, dtype='float32')
#
#         embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100)
#
#         all_embs = np.stack(embeddings_index.values())
#         emb_mean, emb_std = all_embs.mean(), all_embs.std()
#         embed_size = all_embs.shape[1]
#
#         # word_index = tokenizer.word_index
#         nb_words = min(self.max_features, len(word_index))
#         embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#         for word, i in word_index.items():
#             if i >= self.max_features: continue
#             embedding_vector = embeddings_index.get(word)
#             if embedding_vector is not None: embedding_matrix[i] = embedding_vector
#
#         return embedding_matrix
#
#     def load_para(self, word_index):
#         EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
#
#         def get_coefs(word, *arr):
#             return word, np.asarray(arr, dtype='float32')
#
#         embeddings_index = dict(
#             get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)
#
#         all_embs = np.stack(embeddings_index.values())
#         emb_mean, emb_std = -0.0053247833, 0.49346462
#         embed_size = all_embs.shape[1]
#
#         # word_index = tokenizer.word_index
#         nb_words = min(self.max_features, len(word_index))
#         embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#         for word, i in word_index.items():
#             if i >= self.max_features: continue
#             embedding_vector = embeddings_index.get(word)
#             if embedding_vector is not None: embedding_matrix[i] = embedding_vector
#
#         return embedding_matrix
#
#     def compare_lstm_bi_lstm(self, n_time_steps=10):
#
#     # self.lstm_model, self.results = self.do_lstm(n_time_steps=n_time_steps, backwards=False, result_name='forward')
#     # self.lstm_model, self.results = self.do_lstm(n_time_steps=n_time_steps, backwards=True, result_name='backward')
#     # self.bi_lstm_model, self.results = self.do_bi_lstm(n_time_steps=n_time_steps, result_name='backward')
#     #
#     # print(self.results)
#     # self.results.plot()
#     # pyplot.show()
#
#     def do_lstm(self, n_time_steps, backwards, result_name):
#
#     # result_name = 'lstm_' + str(result_name)
#     # self.lstm_model = self.get_lstm_model(n_time_steps=n_time_steps, backwards=backwards)
#     # self.results[result_name] = self.train_model(model=self.lstm_model, n_time_steps=n_time_steps)
#     # return self.lstm_model, self.results
#
#     def do_bi_lstm(self, n_time_steps, result_name, mode='concat'):
#         self.__seed_everything()
#
#     # result_name = 'bi_lstm_' + str(result_name)
#     # self.bi_lstm_model = self.get_bi_lstm_model(n_time_steps=n_time_steps, mode=mode)
#     # self.results[result_name] = self.train_model(model=self.bi_lstm_model, n_time_steps=n_time_steps)
#     # return self.bi_lstm_model, self.results
#
#     def get_lstm_model(self, n_time_steps, backwards, activation='sigmoid', loss='binary_crossentropy',
#                        optimizer='adam'):
#
#     # self.lstm_model = Sequential()
#     # self.lstm_model.add(LSTM(20, input_shape=(n_time_steps, 1), return_sequences=True, go_backwards=backwards))
#     # self.lstm_model.add(TimeDistributed(Dense(1, activation=activation)))
#     # self.lstm_model.compile(loss=loss, optimizer=optimizer)
#     #
#     # print('LSTM MODEL ==> n_time_steps is: ' + str(n_time_steps) + '\tbackwards: ' + str(backwards))
#     # return self.lstm_model
#
#     r"""
#         # Arguments
#             merge_mode: Mode by which outputs of the
#                 forward and backward RNNs will be combined.
#                 One of {'sum', 'mul', 'concat', 'ave', None}.
#                 If None, the outputs will not be combined,
#                 they will be returned as a list.
#     """
#
#     def get_bi_lstm_model(self, n_time_steps, mode, activation='sigmoid', loss='binary_crossentropy',
#                           optimizer='adam'):
#
#     # self.bi_lstm_model = Sequential()
#     # self.bi_lstm_model.add(
#     #     Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_time_steps, 1), merge_mode=mode))
#     # self.bi_lstm_model.add(TimeDistributed(Dense(1, activation=activation)))
#     # self.bi_lstm_model.compile(loss=loss, optimizer=optimizer)
#     # print('BiLSTM MODEL ==> n_time_steps is: ' + str(n_time_steps) + '\tmode: ' + str(mode))
#     # return self.bi_lstm_model
#
#     def train_model(self, model, n_time_steps, epoch=250):
#
#     # loss = list()
#     # for _ in range(epoch):
#     #     X, y = self.generate_random_seq(n_time_steps=n_time_steps)
#     #     hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
#     #     loss.append(hist.history['loss'][0])
#     # return loss
#
#     r"""
#         Generate random sequence:
#             create a sequence of random numbers in [0,1]
#             limit = calculate cut-off value to change class values
#             determine the class outcome for each item in cumulative sequence
#             reshape input and output data to be suitable for LSTMs
#     """
#
#     @staticmethod
#     def generate_random_seq(n_time_steps):
# # limit = n_time_steps / 4.0
# #
# # X = array([random() for _ in range(n_time_steps)])
# # y = array([0 if x < limit else 1 for x in cumsum(X)])
# #
# # X = X.reshape(1, n_time_steps, 1)
# # y = y.reshape(1, n_time_steps, 1)
# #
# # return X, y


import random
import copy
import time
import pandas as pd
import numpy as np
import gc
import re
import torch
# from torchtext import data
# import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from collections import Counter
from textblob import TextBlob
from nltk import word_tokenize

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
# from torchtext.data import Example
from sklearn.metrics import f1_score
# import torchtext
import os

from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer
from unidecode import unidecode

from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from multiprocessing import Pool
from functools import partial
import numpy as np
from sklearn.decomposition import PCA
import torch as t
import torch.nn as nn
import torch.nn.functional as F

embed_size = 300  # how big is each word vector
max_features = 120000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70  # max number of words in a question to use
batch_size = 512  # how many samples to process at once
n_epochs = 5  # how many times to iterate over all samples
n_splits = 5  # Number of K-fold Splits
SEED = 10
debug = 0

loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule

def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        # ALLmight
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def load_fasttext(word_index):
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.0053247833, 0.49346462
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


# def known_contractions(embed):
#     known = []
#     for contract in contraction_mapping:
#         if contract in embed:
#             known.append(contract)
#     return known


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', ]


def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


mispell_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
                "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center',
                'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2',
                'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best',
                'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist',
                'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'}


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def load_and_prec():
    if debug:
        train_df = pd.read_csv("../input/train.csv")[:80000]
        test_df = pd.read_csv("../input/test.csv")[:20000]
    else:
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    # lower the question text
    train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())

    # Clean the text
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

    # Clean numbers
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))

    # Clean spellings
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))

    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values

    # shuffling the data

    np.random.seed(SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]

    return train_X, test_X, train_y, tokenizer.word_index


start = time.time()

x_train, x_test, y_train, word_index = load_and_prec()

print(time.time() - start)

# missing entries in the embedding are set using np.random.normal so we have to seed here too
seed_everything()
if debug:
    paragram_embeddings = np.random.randn(120000, 300)
    glove_embeddings = np.random.randn(120000, 300)
    embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
else:
    glove_embeddings = load_glove(word_index)
    paragram_embeddings = load_para(word_index)
    embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
print(np.shape(embedding_matrix))


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def pytorch_model_run_cv(x_train, y_train, features, x_test, model_obj, feats=False, clip=True):
    seed_everything()
    avg_losses_f = []
    avg_val_losses_f = []
    # matrix for the out-of-fold predictions
    train_preds = np.zeros((len(x_train)))
    # matrix for the predictions on the test set
    test_preds = np.zeros((len(x_test)))
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(x_train, y_train))
    for i, (train_idx, valid_idx) in enumerate(splits):
        seed_everything(i * 1000 + i)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if feats:
            features = np.array(features)
        x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
        if feats:
            kfold_X_features = features[train_idx.astype(int)]
            kfold_X_valid_features = features[valid_idx.astype(int)]
        x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()

        model = copy.deepcopy(model_obj)

        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=0.001)

        ################################################################################################
        scheduler = False
        ###############################################################################################

        train = MyDataset(torch.utils.data.TensorDataset(x_train_fold, y_train_fold))
        valid = MyDataset(torch.utils.data.TensorDataset(x_val_fold, y_val_fold))

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {i + 1}')
        for epoch in range(n_epochs):
            start_time = time.time()
            model.train()

            avg_loss = 0.
            for i, (x_batch, y_batch, index) in enumerate(train_loader):
                if feats:
                    f = kfold_X_features[index]
                    y_pred = model([x_batch, f])
                else:
                    y_pred = model(x_batch)

                if scheduler:
                    scheduler.batch_step()

                # Compute and print loss.
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()

            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros((len(x_test)))

            avg_val_loss = 0.
            for i, (x_batch, y_batch, index) in enumerate(valid_loader):
                if feats:
                    f = kfold_X_valid_features[index]
                    y_pred = model([x_batch, f]).detach()
                else:
                    y_pred = model(x_batch).detach()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[index] = sigmoid(y_pred.cpu().numpy())[:, 0]

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss)
        # predict all samples in the test set batch per batch
        for i, (x_batch,) in enumerate(test_loader):
            if feats:
                f = test_features[i * batch_size:(i + 1) * batch_size]
                y_pred = model([x_batch, f]).detach()
            else:
                y_pred = model(x_batch).detach()

            test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)

    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f), np.average(avg_val_losses_f)))
    return train_preds, test_preds


class BiLSTM(nn.Module):

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))

        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        # print("avg_pool", avg_pool.size())
        # print("max_pool", max_pool.size())
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

    # always call this before training for deterministic results


seed_everything()

x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
train_preds, test_preds = pytorch_model_run_cv(x_train, y_train, None, x_test, BiLSTM(), feats=False, clip=False)


def bestThresshold(y_train, train_preds):
    tmp = [0, 0, 0]  # idx, cur, max
    delta = 0
    for tmp[0] in tqdm(np.arange(0.1, 0.501, 0.01)):
        tmp[1] = f1_score(y_train, np.array(train_preds) > tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
    return delta, tmp[2]


delta, _ = bestThresshold(y_train, train_preds)

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import *
import tensorflow as tf


def model_train_cv(x_train, y_train, nfold, model_obj):
    splits = list(StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED).split(x_train, y_train))
    x_train = x_train
    y_train = np.array(y_train)
    # matrix for the out-of-fold predictions
    train_oof_preds = np.zeros((x_train.shape[0]))
    for i, (train_idx, valid_idx) in enumerate(splits):
        print(f'Fold {i + 1}')
        x_train_fold = x_train[train_idx.astype(int)]
        y_train_fold = y_train[train_idx.astype(int)]
        x_val_fold = x_train[valid_idx.astype(int)]
        y_val_fold = y_train[valid_idx.astype(int)]

        clf = copy.deepcopy(model_obj)
        clf.fit(x_train_fold, y_train_fold, batch_size=512, epochs=5, validation_data=(x_val_fold, y_val_fold))

        valid_preds_fold = clf.predict(x_val_fold)[:, 0]

        # storing OOF predictions
        train_oof_preds[valid_idx] = valid_preds_fold
    return train_oof_preds


# BiDirectional LSTM

def model_lstm_du(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    '''
    Here 64 is the size(dim) of the hidden state vector as well as the output vector. Keeping return_sequence we want the output for the entire sequence. So what is the dimension of output for this layer?
        64*70(maxlen)*2(bidirection concat)
    CuDNNLSTM is fast implementation of LSTM layer in Keras which only runs on GPU
    '''
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = model_lstm_du(embedding_matrix)
train_oof_preds = model_train_cv(x_train, y_train, 5, model)
delta, _ = bestThresshold(y_train, train_oof_preds)
