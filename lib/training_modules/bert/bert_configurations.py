import tensorflow
ENABLE_GPU = True

PREPROCESS_SEQ_LEN = 128
PREPROCESS_BATCH_SIZE = 32
PREPROCESS_BUFFER_SIZE = tensorflow.data.AUTOTUNE

PREPROCESS_DO_SHUFFLING = True

PREPROCESS_IGNORE_EXC_STR = True

PREPROCESS_ONLY_SOURCE_TWEET = True

BERT_TEST_SIZE = 0.15
BERT_TRAIN_SIZE = 0.7
BERT_VAL_SIZE = 0.15
BERT_USE_K_FOLD = True
BERT_K_FOLD = 5

BERT_EPOCHS = 10
BERT_BATCH_SIZE = 16
BERT_DROPOUT_RATE = 0.1
BERT_LEARNING_RATE = 2e-5
BERT_OPTIMIZER_NAME = 'adam'  # sgd, adam, adamax, adadelta, adagrad
BERT_SAVE_MODEL_NAME = 'res9_bert'
BERT_SAVE_MODEL_DIR = './saved_models'
BERT_MODEL_NAME = "distilbert-base-uncased"
# BERT_MODEL_NAME = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
