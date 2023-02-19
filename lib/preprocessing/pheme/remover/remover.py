from abc import ABC


class Remover(ABC):
    def remove_usernames(self, text):
        pass

    def remove_links(self, text):
        pass

    def remove_emails(self, text):
        pass

    def remove_stop_words(self, tokens):
        pass

    def remove_special_characters(self, tokens):
        pass
