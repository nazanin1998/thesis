from abc import ABC

from lib.preprocessing.pheme.bert_embedding.bert_embedding_impl import BertEmbeddingImpl
from lib.preprocessing.pheme.expander.expander_impl import ExpanderImpl
from lib.preprocessing.pheme.remover.remover_impl import RemoverImpl
from lib.preprocessing.pheme.tokenizing.tokenizer_impl import TokenizerImpl
from lib.preprocessing.pheme.word_root.word_root_lemma_impl import WordRootLemmaImpl


class PreProcess(ABC):
    def initialize_modules(self, expander=ExpanderImpl, remover=RemoverImpl, tokenizer=TokenizerImpl,
                           lemma_maker=WordRootLemmaImpl,
                           embedding_maker=BertEmbeddingImpl, ):
        pass

    def get_preprocessed_dataframe(self):
        pass

