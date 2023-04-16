import numpy

from lib.preprocessing.pheme.bert_embedding.bert_embedding_impl import BertEmbeddingImpl
from lib.preprocessing.pheme.expander.expander_impl import ExpanderImpl
from lib.preprocessing.pheme.preprocessing import PreProcess
from lib.preprocessing.pheme.remover.remover_impl import RemoverImpl
from lib.preprocessing.pheme.tokenizing.tokenizer_impl import TokenizerImpl
from lib.preprocessing.pheme.word_root.word_root_lemma_impl import WordRootLemmaImpl
import lib.utils.constants as constants
from lib.preprocessing.str_processing_impl import StrProcessingImpl

from lib.utils.file_dir_handler import FileDirHandler


# //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

class PreProcessImpl(PreProcess):
    def __init__(self, df=None, ):
        print("\n<< PHASE-2 <==> PREPROCESS >>")
        self.__df = df
        self.__current_index = 0
        self.__expander = None
        self.__remover = None
        self.__tokenizer = None
        self.__lemma_maker = None
        self.__embedding_maker = None

        self.__expanded_text = None
        self.__sent_list = None
        self.__marked_text = None
        self.__text_without_username = None
        self.__text_without_links = None
        self.__text_without_emails = None
        self.__tokens = None
        self.__words_roots = None
        self.__tokens_without_sw = None
        self.__tokens_without_sc = None
        self.__ids = None

    def initialize_modules(self, expander=ExpanderImpl, remover=RemoverImpl, tokenizer=TokenizerImpl,
                           lemma_maker=WordRootLemmaImpl,
                           embedding_maker=BertEmbeddingImpl, ):
        self.__expander = expander()
        self.__remover = remover()
        self.__tokenizer = tokenizer()
        self.__lemma_maker = lemma_maker()
        self.__embedding_maker = embedding_maker()

    def get_preprocessed_dataframe(self):
        preprocess_dir = FileDirHandler.read_directories(directory=constants.PHEME_PRE_PROCESS_CSV_DIR)

        if preprocess_dir is None or not preprocess_dir.__contains__(constants.PHEME_PRE_PROCESS_CSV_NAME):
            self.initialize_modules()
            self.preprocess()
        else:
            self.read_preprocessed_csv_dataset()
        self.print_label_statistics(df=self.__df)
        print("<< PHASE-2 <==> PREPROCESS DONE >>\n")

        return self.__df

    @staticmethod
    def get_array_column_by_name(df, col_name):
        arr = df[col_name].to_numpy()
        arr = list([StrProcessingImpl.convert_str_to_array(item) for item in arr])
        return numpy.vstack(arr)
        # return arr.reshape(1, arr.shape[0], arr.shape[1])

    def preprocess(self):
        self.__df['user.description_pre'] = self.__df['user.description'].apply(
            lambda x: self.single_item_preprocess(x))
        FileDirHandler.write_csv_file(path=constants.PHEME_PRE_PROCESS_CSV_PATH, df=self.__df)
        self.__df['text_pre'] = self.__df['text'].apply(
            lambda x: self.single_item_preprocess(x))
        FileDirHandler.write_csv_file(path=constants.PHEME_PRE_PROCESS_CSV_PATH, df=self.__df)

    def single_item_preprocess(self, text):
        self.__current_index += 1
        print(self.__current_index, end=', ')

        self.__expanded_text = self.__expander.expand_contractions(text=text)

        self.__sent_list = self.__tokenizer.text_to_sentence_list(text=self.__expanded_text)

        self.__marked_text = self.__tokenizer.convert_sentences_list_to_masked_text(sentences=self.__sent_list)

        self.__text_without_username = self.__remover.remove_usernames(text=self.__marked_text)
        self.__text_without_links, links = self.__remover.remove_links(text=self.__text_without_username)
        self.__text_without_emails, emails = self.__remover.remove_emails(text=self.__text_without_links)

        self.__tokens = self.__tokenizer.sentence_to_tokens(sentence=self.__text_without_emails)

        # self.__words_roots = self.__lemma_maker.find_batch_words_root(tokens=self.__tokens)

        self.__tokens_without_sw = self.__remover.remove_stop_words(tokens=self.__tokens)
        self.__tokens_without_sc = self.__remover.remove_special_characters(tokens=self.__tokens_without_sw)

        self.__ids = self.__tokenizer.get_ids_from_tokens(tokens=self.__tokens_without_sc)

        self.print_summery()
        if self.__ids is None or self.__ids is numpy.NaN:
            return None
        return self.__ids
        # self.__sentence = self.tokens_to_sentence(tokens=self.__tokens_without_sc, links=links, emails=emails)
        # self.__sentence = self.__sentence.lower()
        # self.embed = self.__embedding_maker.bert_embed(self.__sentence)
        # self.print_summery()
        # if self.embed is None:
        #     return numpy.NaN
        # return self.embed

    def read_preprocessed_csv_dataset(self):
        print("\tRead Preprocessed CSV Dataset ...", end=' ==> ')
        self.__df = FileDirHandler.read_csv_file(path=constants.PHEME_PRE_PROCESS_CSV_PATH)
        print(self.__df.shape)

    @staticmethod
    def print_label_statistics(df, label_name='is_rumour'):
        df_rumor = df[df[label_name] == 0]
        df_non_rumor = df[df[label_name] == 1]

        print('\t' + label_name + ' == 0: ' + str(df_rumor.shape))
        print('\t' + label_name + ' == 1: ' + str(df_non_rumor.shape))

    def print_summery(self):
        print("\n1 ===> EXPAND")
        print(self.__expanded_text)
        print("2 ===> SENTENCE SEGMENTATION")
        print(self.__sent_list)
        print("3 ===> MARKED TEXT")
        print(self.__marked_text)
        print("4 ===> REMOVE USERNAME")
        print(self.__text_without_username)
        print("5 ===> REMOVE LINKS")
        print(self.__text_without_links)
        print("4 ===> REMOVE EMAILS")
        print(self.__text_without_emails)
        print("3 ===> TOKENS")
        print(self.__tokens)
        print("4 ===> WORD ROOT")
        print(self.__words_roots)
        print("5 ===> REMOVE STOP WORD")
        print(self.__tokens_without_sw)
        print("6 ===> REMOVE SPECIAL CHARS")
        print(self.__tokens_without_sc)
        print("7 ===> MATCHED IDS")
        print(len(self.__ids) if isinstance(self.__ids, list) else [])

    @staticmethod
    def tokens_to_sentence(tokens, emails, links):
        if tokens is None:
            return ''
        sentence = ''
        for token in tokens:
            sentence = sentence + ' ' + token
        if emails is not None:
            for email in emails:
                sentence = sentence + ' ' + email
        if links is not None:
            for link in links:
                sentence = sentence + ' ' + link
        sentence = sentence.strip()
        return sentence
    # def remove_redundant_information(self):
    #     print('remove_redundant_information')
    #
    # def convert_accented_characters_to_ASCIIs(self):
    #     print('convert_accented_characters_to_ASCIIs')
    #
    # def noise_removal(self):
    #     print('Noise Removal')
    #
    # def normalization(self):
    #     print('normalization')


def do_preprocess(dataframe):
    pre_process = PreProcessImpl(df=dataframe)
    return pre_process.get_preprocessed_dataframe()
