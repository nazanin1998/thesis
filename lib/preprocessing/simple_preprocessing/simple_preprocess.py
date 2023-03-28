import numpy

from lib.preprocessing.pheme.expander.expander_impl import ExpanderImpl
from lib.preprocessing.pheme.remover.remover_impl import RemoverImpl


class SimplePreprocess:

    def __init__(self, ):
        print("\n<< PHASE-2 <==> SIMPLE PREPROCESS >>")
        self.__current_index = 0
        self.__expander = ExpanderImpl()
        self.__remover = RemoverImpl()

        self.__expanded_text = None
        self.__text_without_username = None
        self.__text_without_links = None
        self.__text_without_emails = None

    def preprocess(self, df, col_names):
        for col_name in col_names:
            df["preprocessed_"+col_name] = df[col_name].apply(
                lambda x: self.single_item_preprocess(x))
        return df

    def single_item_preprocess(self, text):
        self.__current_index += 1
        print(self.__current_index, end=', ')
        print(self.print_summery())

        self.__expanded_text = self.__expander.expand_contractions(text=text)

        self.__text_without_username = self.__remover.remove_usernames(text=self.__expanded_text)
        self.__text_without_links, links = self.__remover.remove_links(text=self.__text_without_username)
        self.__text_without_emails, emails = self.__remover.remove_emails(text=self.__text_without_links)

        if self.__text_without_emails is None or self.__text_without_emails is numpy.NaN:
            return ""
        return self.__text_without_emails

    def print_summery(self):
        print("\n1 ===> EXPAND")
        print(self.__expanded_text)
        print("2 ===> REMOVE USERNAME")
        print(self.__text_without_username)
        print("3 ===> REMOVE LINKS")
        print(self.__text_without_links)
        print("4 ===> REMOVE EMAILS")
        print(self.__text_without_emails)
