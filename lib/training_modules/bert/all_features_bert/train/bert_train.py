from lib.training_modules.base.train.base_train import  get_optimizer_from_conf, get_sparse_categorical_acc_metric, get_sparse_categorical_cross_entropy
import tensorflow
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD, Adamax, Adadelta, Adagrad
from sklearn.metrics._scorer import metric
from transformers import TFAutoModelForSequenceClassification, create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from sklearn.model_selection import KFold
from lib.utils.constants import PHEME_LABEL_SECONDARY_COL_NAME, TRAIN, VALIDATION, TEST
from lib.training_modules.bert.analysis.bert_model_analysis import BertModelAnalysis
from lib.training_modules.bert.bert_configurations import BERT_BATCH_SIZE, BERT_EPOCHS, BERT_EPOCHS_K_FOLD, BERT_K_FOLD, BERT_MODEL_NAME, BERT_USE_K_FOLD, \
    PREPROCESS_DO_SHUFFLING, BERT_LEARNING_RATE, BERT_OPTIMIZER_NAME
from lib.utils.log.logger import log_end_phase, log_line, log_start_phase, log_phase_desc
import keras.backend as K
import tensorflow as tf

class BertTrain:
    def __init__(self, encoded_dataset, tokenizer):
        self.__tokenizer = tokenizer
        self.__encoded_dataset = encoded_dataset
    
        self.__num_labels = len(self.__encoded_dataset[TRAIN].unique(PHEME_LABEL_SECONDARY_COL_NAME))

        self.__loss = get_sparse_categorical_cross_entropy()
        self.__acc_metric = get_sparse_categorical_acc_metric()
        self.__f1_metric = self.f1_score
        self.__optimizer = get_optimizer_from_conf()

        self.__steps_per_epoch = len(encoded_dataset[TRAIN]) // BERT_BATCH_SIZE
        self.__num_train_steps = int(self.__steps_per_epoch * BERT_EPOCHS)
        self.__num_warmup_steps = int(self.__num_train_steps // 10)
        
        if BERT_USE_K_FOLD:
            self.__validation_steps = 0
        else:
            self.__validation_steps = len(encoded_dataset[VALIDATION]) // BERT_BATCH_SIZE
    
    
    def recall(self, y_true, y_pred):
        y_true = K.ones_like(y_true)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (all_positives + K.epsilon())
        return recall

    def precision(self, y_true, y_pred):
        y_true = K.ones_like(y_true)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
   
    def f1_score(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val
    
    
    def start(self):
        log_start_phase(2, 'BERT MODEL STARTED')
        self.log_configuration()
        
        histories, model = self.__do_training()

        tf_test_dataset = self.prepare_ds(model, self.__encoded_dataset[TEST])

        model.summary()

        analyser = BertModelAnalysis(model=model, histories=histories)
        analyser.plot_bert_model()

        train_acc_list, validation_acc_list, train_loss_list, \
            validation_loss_list, validation_acc_mean, validation_loss_mean, \
                validation_acc_max, validation_loss_max, test_loss, test_accuracy = analyser.evaluation(
            test_tensor_dataset=tf_test_dataset)

        analyser.plot_bert_evaluation_metrics(
            train_acc=train_acc_list,
            val_acc=validation_acc_list,
            train_loss=train_loss_list,
            val_loss=validation_loss_list)

        log_end_phase(3, 'BERT ON TWEET TEXT')
        log_line()

    def __do_training(self):
        if BERT_USE_K_FOLD:
            return self.__k_fold_training()
        else:
            return self.__simple_training()
    
    def __k_fold_training(self):
        kf = KFold(n_splits=BERT_K_FOLD, shuffle=True)
        
        histories = []
        fold_index = 0
        
        model = None
        for train_index, test_index in kf.split(self.__encoded_dataset[TRAIN]):
            fold_index += 1

            model = self.__create_comple_model()

            train_ds = self.__encoded_dataset[TRAIN].select(train_index)
            test_ds = self.__encoded_dataset[TRAIN].select(test_index)

            tf_train_dataset = self.prepare_ds(model, train_ds)
            tf_validation_dataset = self.prepare_ds(model, test_ds)

            history = model.fit(
                tf_train_dataset,
                validation_data=tf_validation_dataset,
                epochs=BERT_EPOCHS_K_FOLD,
                batch_size=BERT_BATCH_SIZE,
            )
            
            histories.append(history)
        return histories, model
    
    def __simple_training(self):
        model = self.__create_comple_model()

        tf_train_dataset = self.prepare_ds(model, self.__encoded_dataset[TRAIN])
        tf_validation_dataset = self.prepare_ds(model, self.__encoded_dataset[VALIDATION])
     
        history = self.__fit_model(model, tf_train_dataset, tf_validation_dataset)
        return [history], model
 
    def __create_comple_model(self):
        model = self.create_classifier_model()
        model.compile(
            optimizer=self.__optimizer, 
            loss=self.__loss, 
            metrics=[self.__acc_metric, self.precision, self.recall, self.f1_score]
            )
        return model
    
    def __fit_model(self, model, tf_train_dataset, tf_validation_dataset):

        tensorboard_callback = TensorBoard(log_dir="./text_classification_model_save/logs")
        callbacks = [
            tensorboard_callback,
        ]
        history = model.fit(
            tf_train_dataset,
            validation_data=tf_validation_dataset,
            epochs=BERT_EPOCHS,
            callbacks=callbacks,
            batch_size=BERT_BATCH_SIZE,
            validation_steps=self.__validation_steps,
        )
        return history

    def prepare_ds(self, model, ds):
        return model.prepare_tf_dataset(
            ds,
            shuffle=PREPROCESS_DO_SHUFFLING,
            batch_size=BERT_BATCH_SIZE,
            tokenizer=self.__tokenizer,
        )

    def create_classifier_model(self):
        id2label = {'0': "Rumor", '1': "Non Rumor"}
        label2id = {val: key for key, val in id2label.items()}

        model = TFAutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME, 
            num_labels=self.__num_labels, 
            id2label=id2label, 
            label2id=label2id
        )
        
        return model

    def log_configuration(self):
        log_phase_desc(f'BERT Model               : {BERT_MODEL_NAME}')
        log_phase_desc(f'Do shuffle on splitting  : {PREPROCESS_DO_SHUFFLING}')
        log_phase_desc(f'Bert batch size          : {BERT_BATCH_SIZE}')
        
        log_phase_desc(f'Bert epochs              : {BERT_EPOCHS_K_FOLD if BERT_USE_K_FOLD else BERT_EPOCHS}')
        # log_phase_desc(f'Bert dropout rate        : {BERT_DROPOUT_RATE}')
        log_phase_desc(f'Bert learning rate       : {BERT_LEARNING_RATE}')
        log_phase_desc(f'Bert optimizer           : {BERT_OPTIMIZER_NAME}')
        log_phase_desc(f'Bert loss                : {self.__loss}')
        log_phase_desc(f'Bert metrics             : {self.__acc_metric}, {self.__f1_metric}')
        log_phase_desc(f'Num labels               : {self.__num_labels}')
