# import spacy
import stanza
import torch
from transformers import BertTokenizer, TensorType

from lib.preprocessing.pheme.tokenizing.tokenizer import Tokenizer


class TokenizerPlusImpl(Tokenizer):
    def __init__(self, pretrained_model_name_or_path='sentence-transformers/all-MiniLM-L6-v2', stanza_lang='en',
                 stanza_processors='tokenize'):
        print('TOKENIZER MODULE  ==> Plus Impl (pretrained_model_name_or_path='
              + pretrained_model_name_or_path + ') ,   Use Stanza for sentence splitting')

        self.__ids = None
        self.__attentions_mask = None
        self.__encoded = None
        self.__nlp = stanza.Pipeline(lang=stanza_lang, processors=stanza_processors)
        self.__bertTokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path)

    r"""
       @:parameter text = the sentence to be encoded
       @:parameter add_special_tokens = Add [CLS] and [SEP]
       @:parameter max_length = maximum length of a sentence
       @:parameter pad_to_max_length = Add [PAD]s
       @:parameter return_attention_mask = Generate the attention mask
       @:parameter return_tensors = ask the function to return TensorType includes this enum {PYTORCH, TENSORFLOW, NUMPY, JAX}
       """

    def text_to_sentence_list(self, text):
        doc = self.__nlp(text)
        doc_sents = [sentence.text for sentence in doc.sentences]
        return doc_sents

    def sentence_to_tokens(self, sentence, add_special_tokens=True, max_length=64, pad_to_max_length=False,
                           return_attention_mask=True, return_tensors=TensorType.PYTORCH):
        if sentence is None:
            return None

        if not isinstance(sentence, str):
            raise 'input sentence should be str type'

        self.__encoded = self.__bertTokenizer.encode_plus(
            text=sentence,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            stride=True,
            padding=pad_to_max_length,
            return_attention_mask=return_attention_mask,
            return_tensors=None if return_tensors is TensorType.PYTORCH else return_tensors,
            truncation=True,
        )
        # self.__encoded
        self.__ids = self.__encoded['input_ids']
        self.__attentions_mask = self.__encoded['attention_mask']

        if return_tensors is TensorType.PYTORCH:
            self.__ids = torch.tensor(self.__ids)
            self.__attentions_mask = torch.tensor(self.__attentions_mask)
        return self.__encoded.tokens, self.__ids, self.__attentions_mask,

    def get_ids_from_tokens(self, tokens):
        return self.__ids, self.__attentions_mask,

    def print_summery(self):
        print(self.__encoded)
# import torch
# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# example_text = '[CLS] ' + "The man I've seen." + ' [SEP] ' + 'He bough a gallon of milk.' + ' [SEP] '
# print('sample text is : ' + example_text)
# tokenized_text = tokenizer.tokenize(text=example_text)
#
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokens=tokenized_text)
#
# for tup in zip(tokenized_text, indexed_tokens):
#     print('{:<12}{:>6,}'.format(tup[0], tup[1]))
#
# segments_ids = [1] * len(tokenized_text)
#
# print('segments_ids are ' + str(segments_ids))
# tokens_tensor = torch.tensor([indexed_tokens])
# print(tokens_tensor)
# print('tokens_tensor are ' + str(tokens_tensor))
#
# segments_tensors = torch.tensor([segments_ids])
# print(segments_tensors)
# print('segments_tensors are ' + str(segments_tensors))
#
# model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
#
#
# model.eval()
# with torch.no_grad():
#     outputs = model(tokens_tensor, segments_tensors)
#     hidden_states = outputs[2]
#     print(outputs[0])
#     print(outputs[1])
#     print(outputs[2])
#
#     print(hidden_states)
