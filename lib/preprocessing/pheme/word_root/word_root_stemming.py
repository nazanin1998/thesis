from nltk.stem import PorterStemmer

from lib.preprocessing.pheme.word_root.word_root import WordRoot


class WordRootStemmingImpl(WordRoot):
    def __init__(self):
        print('Stemming Impl of WORD ROOT MODULE is used')

    def find_batch_words_root(self, tokens):
        ps = PorterStemmer()

        stemmed_tokens = []
        for w in tokens:
            value = ps.stem(w)
            stemmed_tokens.append(value)
        return stemmed_tokens
