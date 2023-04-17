import tensorflow

from lib.training_modules.bert.bert_configurations import BERT_OPTIMIZER_NAME, BERT_LEARNING_RATE
from keras.optimizers import Adam, SGD, Adamax, Adadelta, Adagrad


def get_optimizer_from_conf():
    if BERT_OPTIMIZER_NAME == 'adam':
        return Adam(learning_rate=BERT_LEARNING_RATE)
    elif BERT_OPTIMIZER_NAME == "sgd":
        return SGD(learning_rate=BERT_LEARNING_RATE)
    elif BERT_OPTIMIZER_NAME == "adamax":
        return Adamax(learning_rate=BERT_LEARNING_RATE)
    elif BERT_OPTIMIZER_NAME == "adadelta":
        return Adadelta(learning_rate=BERT_LEARNING_RATE)
    elif BERT_OPTIMIZER_NAME == "adagrad":
        return Adagrad(learning_rate=BERT_LEARNING_RATE)

def get_optimizer(self):
    return create_optimizer(
        init_lr=BERT_LEARNING_RATE,
        num_warmup_steps=self.__num_warmup_steps,
        num_train_steps=self.__num_train_steps
    )    

def get_sparse_categorical_acc_metric():
    return tensorflow.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tensorflow.float32)


def get_sparse_categorical_cross_entropy():
    return tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

