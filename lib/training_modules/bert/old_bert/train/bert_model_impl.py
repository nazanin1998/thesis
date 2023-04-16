import os

from lib.training_modules.bert.analysis.bert_model_analysis import BertModelAnalysis
from lib.training_modules.bert.bert_configurations import BERT_DROPOUT_RATE, BERT_BATCH_SIZE, \
    BERT_EPOCHS, BERT_SAVE_MODEL_DIR, BERT_LEARNING_RATE, BERT_SAVE_MODEL_NAME, PREPROCESS_SEQ_LEN, \
    PREPROCESS_BATCH_SIZE, \
    PREPROCESS_BUFFER_SIZE, PREPROCESS_ONLY_SOURCE_TWEET, PREPROCESS_DO_SHUFFLING, BERT_OPTIMIZER_NAME, BERT_MODEL_NAME
from lib.training_modules.bert.bert_model_name import get_bert_model_name
from lib.training_modules.bert.train.bert_model import MyBertModel
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_text as text  # A dependency of the preprocessing model

from lib.utils.log.logger import log_start_phase, log_end_phase, log_line, print_indented_key_value, print_indented, \
    log_phase_desc


class BertModelImpl(MyBertModel):
    def __init__(self,
                 train_tensor_dataset,
                 val_tensor_dataset,
                 test_tensor_dataset,
                 num_classes,
                 train_len,
                 validation_len,
                 test_len,
                 bert_preprocess_model,
                 bert_model_name=BERT_MODEL_NAME):
        self.__train_tensor_dataset = train_tensor_dataset
        self.__val_tensor_dataset = val_tensor_dataset
        self.__test_tensor_dataset = test_tensor_dataset
        self.__num_classes = num_classes
        self.__train_len = train_len
        self.__validation_len = validation_len
        self.__test_len = test_len
        self.__bert_preprocess_model = bert_preprocess_model

        self.bert_model_name = get_bert_model_name(bert_model_name=bert_model_name)

        self.__loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.__metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)

        self.__steps_per_epoch = self.__train_len // BERT_BATCH_SIZE
        self.__validation_steps = self.__validation_len // BERT_BATCH_SIZE
        self.__num_train_steps = self.__steps_per_epoch * BERT_EPOCHS
        self.__num_warmup_steps = self.__num_train_steps // 10

    @staticmethod
    def get_optimizer():
        # optimizer = optimization.create_optimizer(
        #     init_lr=init_lr,
        #     num_train_steps=num_train_steps,
        #     num_warmup_steps=num_warmup_steps,
        #     optimizer_type='adamw')

        if BERT_OPTIMIZER_NAME == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=BERT_LEARNING_RATE)
        elif BERT_OPTIMIZER_NAME == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=BERT_LEARNING_RATE)
        elif BERT_OPTIMIZER_NAME == "adamax":
            return tf.keras.optimizers.Adamax(learning_rate=BERT_LEARNING_RATE)
        elif BERT_OPTIMIZER_NAME == "adadelta":
            return tf.keras.optimizers.Adadelta(learning_rate=BERT_LEARNING_RATE)
        elif BERT_OPTIMIZER_NAME == "adagrad":
            return tf.keras.optimizers.Adagrad(learning_rate=BERT_LEARNING_RATE)


    def __fit_model(self, model):
        history = model.fit(
            x=self.__train_tensor_dataset,
            batch_size=BERT_BATCH_SIZE,
            validation_data=self.__val_tensor_dataset,
            steps_per_epoch=self.__steps_per_epoch,
            epochs=BERT_EPOCHS,
            validation_steps=self.__validation_steps,
        )
        return history

    def start(self):
        log_start_phase(3, 'BERT MODEL STARTED')
        log_phase_desc(f'BERT Model               : {self.bert_model_name}')
        log_phase_desc(f'Preprocess sequence len  : {PREPROCESS_SEQ_LEN}')
        log_phase_desc(f'Preprocess batch size    : {PREPROCESS_BATCH_SIZE}')
        log_phase_desc(f'Preprocess buffer size   : {PREPROCESS_BUFFER_SIZE}')
        log_phase_desc(f'Do shuffle on splitting  : {PREPROCESS_DO_SHUFFLING}')
        log_phase_desc(f'Bert batch size          : {BERT_BATCH_SIZE}')
        log_phase_desc(f'Bert epochs              : {BERT_EPOCHS}')
        log_phase_desc(f'Bert dropout rate        : {BERT_DROPOUT_RATE}')
        log_phase_desc(f'Bert learning rate       : {BERT_LEARNING_RATE}')
        log_phase_desc(f'Bert optimizer           : {BERT_OPTIMIZER_NAME}')
        log_phase_desc(f'Assume only source tweets: {PREPROCESS_ONLY_SOURCE_TWEET}')

        classifier_model = self.build_classifier_model(self.__num_classes)

        classifier_model.compile(
            optimizer=self.get_optimizer(),
            loss=self.__loss, metrics=[self.__metrics])

        history = self.__fit_model(classifier_model)

        classifier_model.summary()

        self.save_model(classifier_model=classifier_model)

        analyser = BertModelAnalysis(model=classifier_model, history=history)
        analyser.plot_bert_model()
        acc, val_acc, loss, val_loss, first_loss, accuracy = analyser.evaluation(
            test_tensor_dataset=self.__test_tensor_dataset)

        analyser.plot_bert_evaluation_metrics(
            acc=acc,
            val_acc=val_acc,
            loss=loss,
            val_loss=val_loss)

        log_end_phase(3, 'BERT ON TWEET TEXT')
        log_line()

    @staticmethod
    def get_save_model_path():
        return os.path.join(BERT_SAVE_MODEL_DIR, BERT_SAVE_MODEL_NAME)

    def save_model(self, classifier_model):
        saved_model_path = self.get_save_model_path()
        # preprocess_inputs = self.__bert_preprocess_model.inputs
        # bert_encoder_inputs = self.__bert_preprocess_model(preprocess_inputs)
        # bert_outputs = classifier_model(bert_encoder_inputs)

        # model_for_export = tf.keras.Model(preprocess_inputs, bert_outputs)
        classifier_model.save(saved_model_path, include_optimizer=True)
        # model_for_export.save(saved_model_path, include_optimizer=True)
        print(f'SAVE MODEL (PATH): {saved_model_path}')

    def build_classifier_model(self_out, num_classes):
        class Classifier(tf.keras.Model):
            def __init__(self, num_classes):
                super(Classifier, self).__init__(name="prediction")
                self.encoder = hub.KerasLayer(self_out.bert_model_name, trainable=True)
                self.dropout = tf.keras.layers.Dropout(BERT_DROPOUT_RATE)
                self.dense = tf.keras.layers.Dense(num_classes)

            def call(self, preprocessed_text):
                encoder_outputs = self.encoder(preprocessed_text)
                pooled_output = encoder_outputs["pooled_output"]
                x = self.dropout(pooled_output)
                x = self.dense(x)
                return x

        model = Classifier(num_classes)

        return model
