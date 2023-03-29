import contractions
import numpy

from lib.preprocessing.pheme.expander.expander import Expander


class ExpanderImpl(Expander):
    def __init__(self):
        print('\tEXPANDER MODULE ==> Initiate')

    def expand_contractions(self, text):
        expanded_words = []
        if text is numpy.NaN or text is None:
            return None
        for word in text.split():
            try:
                expanded_words.append(contractions.fix(word))
            except:
                break
        text = ' '.join(expanded_words)
        return text
