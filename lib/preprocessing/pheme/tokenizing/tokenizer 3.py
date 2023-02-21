from abc import ABC


class Tokenizer(ABC):

    def text_to_sentence_list(self, text):
        pass

    def convert_sentences_list_to_masked_text(self, sentences):
        pass

    def sentence_to_tokens(self, sentence):
        pass

    def complete_tokenizing(self, text):
        pass

    def get_ids_from_tokens(self, tokens):
        pass
