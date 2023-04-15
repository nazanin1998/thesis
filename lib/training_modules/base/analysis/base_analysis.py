from lib.training_modules.bert.bert_configurations import BERT_MODEL_NAME, PREPROCESS_DO_SHUFFLING, BERT_BATCH_SIZE, \
    BERT_EPOCHS, BERT_LEARNING_RATE, BERT_OPTIMIZER_NAME
from lib.utils.log.logger import log_phase_desc


def log_configurations():
    log_phase_desc(f'BERT Model               : {BERT_MODEL_NAME}')
    # log_phase_desc(f'Preprocess sequence len  : {PREPROCESS_SEQ_LEN}')
    # log_phase_desc(f'Preprocess batch size    : {PREPROCESS_BATCH_SIZE}')
    # log_phase_desc(f'Preprocess buffer size   : {PREPROCESS_BUFFER_SIZE}')
    log_phase_desc(f'Do shuffle on splitting  : {PREPROCESS_DO_SHUFFLING}')
    log_phase_desc(f'Bert batch size          : {BERT_BATCH_SIZE}')
    log_phase_desc(f'Bert epochs              : {BERT_EPOCHS}')
    # log_phase_desc(f'Bert dropout rate        : {BERT_DROPOUT_RATE}')
    log_phase_desc(f'Bert learning rate       : {BERT_LEARNING_RATE}')
    log_phase_desc(f'Bert optimizer           : {BERT_OPTIMIZER_NAME}')
    # log_phase_desc(f'Assume only source tweets: {PREPROCESS_ONLY_SOURCE_TWEET}')


def get_history_metrics(hist):
    history_dict = hist.history

    train_loss = history_dict['loss']
    validation_loss = history_dict['val_loss']
    train_acc = history_dict['accuracy']
    validation_acc = history_dict['val_accuracy']

    return train_loss, validation_loss, train_acc, validation_acc