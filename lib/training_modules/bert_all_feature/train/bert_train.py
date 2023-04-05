import tensorflow
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from tensorflow.python.keras.callbacks import History
from transformers import TFAutoModelForSequenceClassification, create_optimizer

from lib.training_modules.base.analysis.base_analysis import log_configurations, get_history_metrics
from lib.training_modules.base.train.base_train import get_optimizer_from_conf, get_sparse_categorical_acc_metric, \
    get_sparse_categorical_cross_entropy
from lib.utils.constants import TRAIN, VALIDATION, TEST, PHEME_LABEL_SECONDARY_COL_NAME
from lib.training_modules.bert.analysis.bert_model_analysis import BertModelAnalysis
from lib.training_modules.bert.bert_configurations import BERT_BATCH_SIZE, BERT_EPOCHS, BERT_MODEL_NAME, \
    PREPROCESS_DO_SHUFFLING, BERT_LEARNING_RATE, BERT_OPTIMIZER_NAME, BERT_USE_K_FOLD, BERT_K_FOLD, BERT_EPOCHS_K_FOLD
from lib.utils.log.logger import log_end_phase, log_line, log_start_phase, log_phase_desc


class BertTrain:
    def __init__(self, encoded_dataset, tokenizer):
        self.__id2label = {'0': "Rumor", '1': "Non Rumor"}
        self.__label2id = {val: key for key, val in self.__id2label.items()}

        self.__tokenizer = tokenizer
        self.__encoded_dataset = encoded_dataset

        self.__num_labels = len(self.__encoded_dataset[TRAIN].unique(PHEME_LABEL_SECONDARY_COL_NAME))

        self.__loss = get_sparse_categorical_cross_entropy()
        self.__metrics = get_sparse_categorical_acc_metric()
        self.__optimizer = get_optimizer_from_conf()

        self.__steps_per_epoch = len(encoded_dataset[TRAIN]) // BERT_BATCH_SIZE
        self.__num_train_steps = int(self.__steps_per_epoch * BERT_EPOCHS)
        self.__num_warmup_steps = int(self.__num_train_steps // 10)

        if not BERT_USE_K_FOLD:
            self.__validation_steps = len(encoded_dataset[VALIDATION]) // BERT_BATCH_SIZE

    def start(self):
        log_start_phase(1, 'BERT MODEL STARTED')
        log_configurations()

        history, model = self.__fit_model()

        analyser = BertModelAnalysis(model=model, history=history)
        analyser.plot_bert_model()
        if BERT_USE_K_FOLD:
            train_acc, validation_acc, train_loss, validation_loss = analyser.evaluation()
        else:
            tf_test_dataset = self.prepare_special_ds_for_model(model, self.__encoded_dataset[TEST])

            train_acc, validation_acc, train_loss, validation_loss, test_loss, test_accuracy = analyser.evaluation(
                test_tensor_dataset=tf_test_dataset)

        analyser.plot_bert_evaluation_metrics(
            train_acc=train_acc,
            val_acc=validation_acc,
            train_loss=train_loss,
            val_loss=validation_loss)

        log_end_phase(3, 'BERT ON TWEET TEXT')
        log_line()

    def get_optimizer(self):
        return create_optimizer(
            init_lr=BERT_LEARNING_RATE,
            num_warmup_steps=self.__num_warmup_steps,
            num_train_steps=self.__num_train_steps
        )

    def __fit_model(self):
        if BERT_USE_K_FOLD:
            return self.__k_fold_fit()
        else:
            return self.__simple_fit()

    def __simple_fit(self):

        model = self.__create_compile_model()
        print("baba")
        tf_train_dataset = self.prepare_special_ds_for_model(model, self.__encoded_dataset[TRAIN])
        print("baba1")
        tf_validation_dataset = self.prepare_special_ds_for_model(model, self.__encoded_dataset[VALIDATION])
        print("baba2")

        history = model.fit(
            tf_train_dataset,
            validation_data=tf_validation_dataset,
            epochs=BERT_EPOCHS,
            callbacks=[],
            batch_size=BERT_BATCH_SIZE,
            validation_steps=self.__validation_steps,
        )
        return history, model

    def __k_fold_fit(self):
        kf = KFold(n_splits=BERT_K_FOLD, shuffle=True)

        histories = []
        fold_index = 0

        model = None
        for train_index, test_index in kf.split(self.__encoded_dataset[TRAIN]):
            fold_index += 1

            model = self.__create_compile_model()

            train_ds = self.__encoded_dataset[TRAIN].select(train_index)
            test_ds = self.__encoded_dataset[TRAIN].select(test_index)

            tf_train_dataset = self.prepare_special_ds_for_model(model, train_ds)
            tf_validation_dataset = self.prepare_special_ds_for_model(model, test_ds)

            history = model.fit(
                tf_train_dataset,
                validation_data=tf_validation_dataset,
                epochs=BERT_EPOCHS_K_FOLD,
                callbacks=[],

                batch_size=BERT_BATCH_SIZE,
                # validation_steps=self.__validation_steps,
            )
            histories.append(history)
        return histories, model

    def __create_compile_model(self):
        print('befor model create')
        model = self.create_bert_classifier_model()
        print('after model create')
        model.compile(optimizer=Adam(learning_rate=BERT_LEARNING_RATE), loss=self.__loss, metrics=[self.__metrics])
        print('after model compile')
        return model

    def prepare_special_ds_for_model(self, model, ds):
        return model.prepare_tf_dataset(
            ds,
            shuffle=PREPROCESS_DO_SHUFFLING,
            batch_size=BERT_BATCH_SIZE,
            tokenizer=self.__tokenizer,
        )

    @staticmethod
    def create_bert_classifier_model():
        # print(f'num labels {self.__num_labels}')
        # print(f'num id2label {self.__id2label}')
        # print(f'num lable {self.__label2id}')
        # model = TFAutoModelForSequenceClassification.from_pretrained(
        #     BERT_MODEL_NAME,
        #     num_labels=self.__num_labels,
        #     id2label=self.__id2label,
        #     label2id=self.__label2id
        # )
        id2label = {'0': "Rumor", '1': "Non Rumor"}
        label2id = {val: key for key, val in id2label.items()}
        model = TFAutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
        )
        # print(f'num lable {self.__label2id}')
        return model
