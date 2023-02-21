from lib.preprocessing.pheme.bert_embedding.bert_embedding import BertEmbedding

from sentence_transformers import SentenceTransformer


class BertEmbeddingImpl(BertEmbedding):
    def __init__(self, bert_pretrained_model_name='all-MiniLM-L6-v2'):
        print(
            '\tBERT EMBEDDING MODULE  ==> Initiate SentenceTransformer Impl (bert_pretrained_model_name='
            + bert_pretrained_model_name + ')')

        self.model = SentenceTransformer(bert_pretrained_model_name)

    def bert_embed(self, text):
        embeddings = self.model.encode(text)

        return embeddings

    def word_bert_embed(self, word):
        embeddings = self.model.encode(word, output_value='token_embeddings', convert_to_numpy=True)
        return embeddings

    def sentence_bert_embed(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings
