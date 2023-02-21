import numpy
from stanza import DownloadMethod
from transformers import BertTokenizer
import stanza

from lib.preprocessing.pheme.tokenizing.tokenizer import Tokenizer
from spacy.lang.en import English


class TokenizerImpl(Tokenizer):
    r"""
       Log level for stanza includes
       all_levels = ['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'CRITICAL', 'FATAL']
    """

    def __init__(self, use_vocab_file=False, pretrained_model_name_or_path='bert-base-uncased',
                 vocab_file_path='./vocab.txt', stanza_lang='en', stanza_processors='tokenize',
                 tokenize_no_ssplit=False, stanza_log_level='FATAL'):
        print('\tTOKENIZER MODULE  ==> Initiate (pretrained_model_name_or_path='
              + pretrained_model_name_or_path + '), Use Stanza for sentence segmentation')

        self.__ids = None
        self.__tokens = None

        # self.__nlp = stanza.Pipeline(lang=stanza_lang, processors=stanza_processors,
        #                              tokenize_no_ssplit=tokenize_no_ssplit, tokenize_pretokenized=False, verbose=False,
        #                              logging_level=None, download_method=DownloadMethod.REUSE_RESOURCES)

        self.__nlp = English()
        self.__nlp.add_pipe("sentencizer")

        if use_vocab_file:
            self.__bertTokenizer = BertTokenizer(vocab_file=vocab_file_path)
        else:
            self.__bertTokenizer = BertTokenizer.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path)

    r"""
       This function throw Exception if text is not masked. 
       Mask means , sentence should seperate with [CLS], [SEP], [PAD],...
       EXAMPLE = '[CLS] ' + "The man I've seen." + ' [SEP] ' + 'He bough a gallon of milk.' + ' [SEP] '
       """

    def text_to_sentence_list(self, text):
        if text is None or text is numpy.NaN:
            return None
        doc = self.__nlp(text)
        print([sent.text.strip() for sent in doc.sents])
        return [sent.text.strip() for sent in doc.sents]

    def sentence_to_tokens(self, sentence):
        if sentence is None or sentence is numpy.NaN:
            return None

        return self.__get_tokens_from_masked_text(masked_text=sentence)
        pass

    def convert_sentences_list_to_masked_text(self, sentences):
        if sentences is None or sentences is numpy.NaN:
            return None
        if not isinstance(sentences, list):
            raise Exception("Input sentences must be list type")
        flat_sentence = ''
        for idx, sentence in enumerate(sentences):
            if idx == 0:
                flat_sentence = '[CLS] ' + sentence
                if idx == len(sentences) - 1:
                    flat_sentence = flat_sentence + ' [SEP]'
            else:
                flat_sentence = flat_sentence + ' [SEP] ' + sentence
        if flat_sentence == '':
            return '[PAD]'
        return flat_sentence

    def __get_tokens_from_masked_text(self, masked_text):
        if '[CLS]' not in masked_text and '[SEP]' not in masked_text and '[PAD]' not in masked_text:
            self.__tokens = None
            raise Exception("Input text is not masked")

        self.__tokens = self.__bertTokenizer.tokenize(text=masked_text)
        return self.__tokens

    def complete_tokenizing(self, text):
        sent_list = self.text_to_sentence_list(text=text)
        marked_text = self.convert_sentences_list_to_masked_text(sentences=sent_list)
        tokens = self.sentence_to_tokens(sentence=marked_text)
        return tokens

    def get_ids_from_tokens(self, tokens, max_len=64):
        if tokens is None or tokens is numpy.NaN:
            return None
        self.__ids = self.__bertTokenizer.convert_tokens_to_ids(tokens=tokens)

        if len(self.__ids) != 64:
            pad_id = self.__bertTokenizer.convert_tokens_to_ids(tokens='[PAD]')
            self.__ids = pad(my_list=self.__ids, content=pad_id, width=64)
        return self.__ids

    def print_summery(self):
        for tup in zip(self.__tokens, self.__ids):
            print('{:<12}{:>6,}'.format(tup[0], tup[1]))


def pad(my_list, content, width):
    my_list.extend([content] * (width - len(my_list)))
    return my_list
