from nltk import WordNetLemmatizer
from lib.preprocessing.pheme.word_root.word_root import WordRoot


class WordRootLemmaImpl(WordRoot):

    def __init__(self):
        print('\tWORD ROOT MODULE ==> Initiate Lemma Impl')
        self.wordnet_lemma_maker = WordNetLemmatizer()

    def find_batch_words_root(self, tokens):
        if tokens is None:
            return None

        lemma_tokens = []
        for w in tokens:
            value = self.wordnet_lemma_maker.lemmatize(w, pos="v")
            lemma_tokens.append(value)
        return lemma_tokens
