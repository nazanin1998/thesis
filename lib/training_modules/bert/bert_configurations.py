import tensorflow

preprocess_seq_length = 128
preprocess_batch_size = 128
preprocess_buffer_size = tensorflow.data.AUTOTUNE

r"""
    In preprocessing we need to make input for each column of ds.
    By this parameter we are able to just consider text columns as input for bert.
    """
preprocess_ignore_exc_str = True

init_lr = 2e-5

bert_epochs = 1
bert_dropout_rate = 0.1

bert_test_size = 0.2
bert_train_size = 0.6
bert_val_size = 0.2
