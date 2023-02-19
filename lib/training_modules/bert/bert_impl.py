import torch
from transformers import BertTokenizer, BertModel
from lib.training_modules.bert.bert import Bert


class BertImpl(Bert):
    def __init__(self, pretrained_model_name_or_path='bert-base-uncased', ):
        self.ids = None
        self.tokens = None
        self.bertTokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)


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

    def print_summery(self):
        for tup in zip(self.tokens, self.ids):
            print('{:<12}{:>6,}'.format(tup[0], tup[1]))


#tz = BertTokenizer.from_pretrained("bert-base-cased")

# # The senetence to be encoded
# sent = "Let's learn deep learning!"
#
# # Encode the sentence
# encoded = tz.encode_plus(
#     text=sent,  # the sentence to be encoded
#     add_special_tokens=True,  # Add [CLS] and [SEP]
#     max_length = 64,  # maximum length of a sentence
#     pad_to_max_length=True,  # Add [PAD]s
#     return_attention_mask = True,  # Generate the attention mask
#     return_tensors = 'pt',  # ask the function to return PyTorch tensors
# )
#





# # Get the input IDs and attention mask in tensor format
# input_ids = encoded['input_ids']
# attn_mask = encoded['attention_mask']

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# example_text = '[CLS] ' + "The man I've seen." + ' [SEP] ' + 'He bough a gallon of milk.' + ' [SEP] '
# print('sample text is : ' + example_text)
# tokenized_text = tokenizer.tokenize(text=example_text)
#
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokens=tokenized_text)

# for tup in zip(tokenized_text, indexed_tokens):
#     print('{:<12}{:>6,}'.format(tup[0], tup[1]))

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

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

model.eval()
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
    print(outputs[0])
    print(outputs[1])
    print(outputs[2])

    print(hidden_states)
