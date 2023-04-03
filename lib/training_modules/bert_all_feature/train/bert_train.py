import tensorflow
from keras.callbacks import TensorBoard
from sklearn.model_selection import KFold
from tensorflow.python.keras.callbacks import History
from transformers import TFAutoModelForSequenceClassification, create_optimizer

from lib.training_modules.base.analysis.base_analysis import log_configurations, get_history_metrics
from lib.training_modules.base.train.base_train import get_optimizer_from_conf, get_sparse_categorical_acc_metric, \
    get_sparse_categorical_cross_entropy
from lib.utils.constants import TRAIN, VALIDATION, TEST, PHEME_LABEL_SECONDARY_COL_NAME
from lib.training_modules.bert.analysis.bert_model_analysis import BertModelAnalysis
from lib.training_modules.bert.bert_configurations import BERT_BATCH_SIZE, BERT_EPOCHS, BERT_MODEL_NAME, \
    PREPROCESS_DO_SHUFFLING, BERT_LEARNING_RATE, BERT_OPTIMIZER_NAME, BERT_USE_K_FOLD, BERT_K_FOLD
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

        # def compute_metrics(predictions, labels):
        #     decoded_predictions = self.__tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #     decoded_labels = self.__tokenizer.batch_decode(labels, skip_special_tokens=True)
        #     result = metric.compute(predictions=decoded_predictions, references=decoded_labels)
        #     return {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # def compute_metrics(eval_predictions):
        #
        #     predictions, labels = eval_predictions
        #     if self.__task != "stsb":
        #         predictions = np.argmax(predictions, axis=1)
        #     else:
        #         predictions = predictions[:, 0]
        #     return metric.compute(predictions=predictions, references=labels)

        history = self.__fit_model()

        model.summary()

        analyser = BertModelAnalysis(model=model, history=history)
        analyser.plot_bert_model()
        # acc, val_acc, loss, val_loss, first_loss, accuracy \
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

        model_name = BERT_MODEL_NAME.split("/")[-1]

        tensorboard_callback = TensorBoard(log_dir="./text_classification_model_save/logs")

        callbacks = [tensorboard_callback]

        if BERT_USE_K_FOLD:
            kf = KFold(n_splits=BERT_K_FOLD, shuffle=True)

            results = []
            fold_index = 0
            for train_index, test_index in kf.split(self.__encoded_dataset[TRAIN]):
                fold_index+=1
                print(f"splits=> train_index: {train_index}, val_index: {test_index}")
                print(f"splits=> train_index_len: {len(train_index)}, val_index_len: {len(test_index)}")
                # # splitting Dataframe (dataset not included)

                model = self.__create_compile_model()

                train_ds = self.__encoded_dataset[TRAIN].select(train_index)
                test_ds = self.__encoded_dataset[TRAIN].select(test_index)

                tf_train_dataset = self.prepare_special_ds_for_model(model, train_ds)
                tf_validation_dataset = self.prepare_special_ds_for_model(model, test_ds)

                history = model.fit(
                    tf_train_dataset,
                    validation_data=tf_validation_dataset,
                    epochs=1,
                    callbacks=callbacks,
                    batch_size=BERT_BATCH_SIZE,
                    # validation_steps=self.__validation_steps,
                )
                # model.eval_model(test_ds)
                # model.train_model(train_df)
                # # validate the model
                # result, model_outputs, wrong_predictions = model.eval_model(val_df, acc=accuracy_score)
                # print(result['acc'])
                # # append model score
                train_loss, validation_loss, train_acc, validation_acc = get_history_metrics(history)

                log_phase_desc(f'FOLD {fold_index}')
                log_phase_desc(f'Training   => Accuracy: {train_acc}, Loss: {train_loss}')
                log_phase_desc(f'Test       => Accuracy: {validation_acc}, Loss: {validation_loss}')

                results.append(history)

            print("results", results)
            # print(f"Mean-Precision: {sum(results) / len(results)}")

            return results
        else:
            model = self.__create_compile_model()

            tf_train_dataset = self.prepare_special_ds_for_model(model, self.__encoded_dataset[TRAIN])
            tf_validation_dataset = self.prepare_special_ds_for_model(model, self.__encoded_dataset[VALIDATION])

            history = model.fit(
                tf_train_dataset,
                validation_data=tf_validation_dataset,
                epochs=BERT_EPOCHS,
                callbacks=callbacks,
                batch_size=BERT_BATCH_SIZE,
                validation_steps=self.__validation_steps,
            )
            return history

    def __create_compile_model(self):
        model = self.create_bert_classifier_model()
        model.compile(optimizer=self.__optimizer, loss=self.__loss, metrics=[self.__metrics])
        return model

    def do_k_fold_fit(self):
        n = 5
        kf = KFold(n_splits=BERT_K_FOLD, random_state=None, shuffle=True)

        results = []

        for train_index, val_index in kf.split(self.__encoded_dataset[TRAIN]):
            # splitting Dataframe (dataset not included)
            train_df = self.__encoded_dataset[TRAIN].iloc[train_index]
            val_df = self.__encoded_dataset[TRAIN].iloc[val_index]
            # Defining Model
            model = ClassificationModel('bert', 'bert-base-uncased')
            # train the model
            model.train_model(train_df)
            # validate the model
            result, model_outputs, wrong_predictions = model.eval_model(val_df, acc=accuracy_score)
            print(result['acc'])
            # append model score
            results.append(result['acc'])

    def prepare_special_ds_for_model(self, model, ds):
        return model.prepare_tf_dataset(
            ds,
            shuffle=PREPROCESS_DO_SHUFFLING,
            batch_size=BERT_BATCH_SIZE,
            tokenizer=self.__tokenizer,
        )

    def create_bert_classifier_model(self):

        model = TFAutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME,
            num_labels=self.__num_labels,
            id2label=self.__id2label,
            label2id=self.__label2id
        )
        return model
