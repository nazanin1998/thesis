from abc import ABC


class BertEmbedding(ABC):
    def bert_embed(self, text):
        pass

    def word_bert_embed(self, word):
        pass

    def sentence_bert_embed(self, sentences):
        pass
