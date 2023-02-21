import tensorflow

preprocess_seq_length = 128
preprocess_batch_size = 32
bert_batch_size = 32
preprocess_buffer_size = tensorflow.data.AUTOTUNE

r"""
    In preprocessing we need to make input for each column of ds.
    By this parameter we are able to just consider text columns as input for bert.
    """
preprocess_ignore_exc_str = True
only_source_tweet = False
save_bert_model_dir = './saved_models'
save_bert_model_name = 'res5_bert'
init_lr = 3e-5
bert_optimizer = 'sgd'  # sgd, adam

bert_epochs = 1
bert_dropout_rate = 0.1
shuffle_data_splitting = False

bert_test_size = 0.2
bert_train_size = 0.75
bert_val_size = 0.05
