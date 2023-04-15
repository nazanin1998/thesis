import numpy

from lib.preprocessing.pheme.bert_embedding.bert_embedding_impl import BertEmbeddingImpl
from lib.preprocessing.pheme.expander.expander_impl import ExpanderImpl
from lib.preprocessing.pheme.preprocessing import PreProcess
from lib.preprocessing.pheme.remover.remover_impl import RemoverImpl
from lib.preprocessing.pheme.tokenizing.tokenizer_plus_impl import TokenizerPlusImpl
from lib.preprocessing.pheme.word_root.word_root_lemma_impl import WordRootLemmaImpl
import lib.utils.constants as constants

from lib.utils.file_dir_handler import FileDirHandler

class PreProcessBertImpl(PreProcess):
    def __init__(self, df, expander=ExpanderImpl(), remover=RemoverImpl(), tokenizer=TokenizerPlusImpl(),
                 lemma_maker=WordRootLemmaImpl(),
                 embedding_maker=BertEmbeddingImpl(), ):
        print("\n<< PHASE-2 <===> PREPROCESS >>")
        self.__df = df
        self.__current_index = 0
        self.__expander = None
        self.__remover = None
        self.__tokenizer = None
        self.__lemma_maker = None
        self.__embedding_maker = None
        self.__preprocess_csv_path = constants.PHEME_PRE_PROCESS_BERT_CSV_NAME
        self.initialize_modules(expander=expander, remover=remover, tokenizer=tokenizer, lemma_maker=lemma_maker,
                                embedding_maker=embedding_maker)

    def initialize_modules(self, expander=ExpanderImpl(), remover=RemoverImpl(), tokenizer=TokenizerPlusImpl(),
                           lemma_maker=WordRootLemmaImpl(),
                           embedding_maker=BertEmbeddingImpl(), ):
        self.__expander = expander
        self.__remover = remover
        self.__tokenizer = tokenizer
        self.__lemma_maker = lemma_maker
        self.__embedding_maker = embedding_maker

    def get_preprocessed_dataframe(self):
        preprocess_dir = FileDirHandler.read_directories(directory=self.__preprocess_csv_path)

        if preprocess_dir is None or not preprocess_dir.__contains__(self.__preprocess_csv_path):
            self.preprocess()
        else:
            self.read_preprocessed_csv_dataset()
        print("<< PHASE-2 <===> PREPROCESS DONE>>")
        return self.__df

    def preprocess(self):
        self.__df['user.description'] = self.__df['user.description'].apply(
            lambda x: self.__preprocess(x))
        FileDirHandler.write_csv_file(path=self.__preprocess_csv_path, df=self.__df)

        self.__df['text'] = self.__df['text'].apply(
            lambda x: self.__preprocess(x))
        FileDirHandler.write_csv_file(path=self.__preprocess_csv_path, df=self.__df)

    def __preprocess(self, text):
        self.__current_index += 1
        print(self.__current_index)
        self.__expanded_text = self.__expander.expand(text=text)

        self.__text_without_username = self.__remover.remove_usernames(text=self.__expanded_text)
        self.__text_without_links, links = self.__remover.remove_links(text=self.__text_without_username)
        self.__text_without_emails, emails = self.__remover.remove_emails(text=self.__text_without_links)

        self.__tokens = self.__tokenizer.tokenize(sentence=self.__text_without_emails)

        self.__words_roots = self.__lemma_maker.find_batch_words_root(tokens=self.__tokens)

        self.__tokens_without_sw = self.__remover.remove_stop_words(tokens=self.__words_roots)

        self.__tokens_without_sc = self.__remover.remove_special_characters(tokens=self.__tokens_without_sw)

        self.__sentence = self.__tokens_to_sentence(tokens=self.__tokens_without_sc, links=links, emails=emails)
        self.__sentence = self.__sentence.lower()
        self.__embed = self.__embedding_maker.bert_embed(self.__sentence)
        # self.print_summery()
        if self.__embed is None:
            return numpy.NaN
        return self.__embed

    def read_preprocessed_csv_dataset(self):
        print("Read Preprocessed CSV Dataset ...", end=' => ')
        self.__df = FileDirHandler.read_csv_file(path=constants.PHEME_PRE_PROCESS_CSV_PATH)
        print(self.__df.shape)

    @staticmethod
    def __tokens_to_sentence(tokens, emails, links):
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

    def print_summery(self):
        print("\n1 => expanded_text")
        print(self.__expanded_text)
        # print("2 => text_without_username")
        # print(self.__text_without_username)
        # print("3 => tokens")
        # print(self.__tokens)
        # print("4 => words_roots")
        # print(self.__words_roots)
        # print("5 => tokens_without_sw")
        # print(self.__tokens_without_sw)
        # print("6 => tokens_without_sc")
        # print(self.__tokens_without_sc)
        print("7 => sentence")
        print(self.__sentence)

    # def __remove_redundant_information(self):
    #     print('remove_redundant_information')
    # 
    # def __convert_accented_characters_to_ASCIIs(self):
    #     print('convert_accented_characters_to_ASCIIs')
    # 
    # def __noise_removal(self):
    #     print('Noise Removal')
    # 
    # def __normalization(self):
    #     print('normalization')
