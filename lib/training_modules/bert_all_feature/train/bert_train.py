import tensorflow
from keras.callbacks import TensorBoard
from transformers import TFAutoModelForSequenceClassification, create_optimizer
from lib.constants import TRAIN, VALIDATION, TEST
from lib.training_modules.bert.analysis.bert_model_analysis import BertModelAnalysis
from lib.training_modules.bert.bert_configurations import BERT_BATCH_SIZE, BERT_EPOCHS, BERT_MODEL_NAME, \
    PREPROCESS_DO_SHUFFLING, BERT_LEARNING_RATE
from lib.utils.log.logger import log_end_phase, log_line


class BertTrain:
    def __init__(self, encoded_dataset, tokenizer):
        self.__tokenizer = tokenizer
        self.__encoded_dataset = encoded_dataset
        self.__loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.__metrics = tensorflow.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tensorflow.float32)

        self.__steps_per_epoch = len(encoded_dataset[TRAIN]) // BERT_BATCH_SIZE
        self.__validation_steps = len(encoded_dataset[VALIDATION]) // BERT_BATCH_SIZE
        self.__num_train_steps = int(self.__steps_per_epoch * BERT_EPOCHS)
        self.__num_warmup_steps = self.__num_train_steps // 10

    def start(self):
        # batches_per_epoch = len(self.__encoded_dataset[TRAIN]) // BERT_BATCH_SIZE
        # total_train_steps = int(batches_per_epoch * BERT_EPOCHS)
        # actual_task = "mnli" if self.__task == "mnli-mm" else self.__task
        # metric = load("glue", self.__task)

        model = self.create_classifier_model()

        tf_train_dataset = self.prepare_ds(model, self.__encoded_dataset[TRAIN])
        tf_validation_dataset = self.prepare_ds(model, self.__encoded_dataset[VALIDATION])
        tf_test_dataset = self.prepare_ds(model, self.__encoded_dataset[TEST])

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
        #     print(f"baby label {labels}")
        #     return metric.compute(predictions=predictions, references=labels)

        optimizer, schedule = self.get_optimizer()

        model.compile(optimizer=optimizer, loss=self.__loss, metrics=[self.__metrics])

        history = self.__fit_model(model, tf_train_dataset, tf_validation_dataset)

        model.summary()

        analyser = BertModelAnalysis(model=model, history=history)
        analyser.plot_bert_model()
        acc, val_acc, loss, val_loss, first_loss, accuracy = analyser.evaluation(
            test_tensor_dataset=tf_test_dataset)

        analyser.plot_bert_evaluation_metrics(
            acc=acc,
            val_acc=val_acc,
            loss=loss,
            val_loss=val_loss)

        log_end_phase(3, 'BERT ON TWEET TEXT')
        log_line()

    def get_optimizer(self):
        return create_optimizer(
            init_lr=BERT_LEARNING_RATE,
            num_warmup_steps=self.__num_warmup_steps,
            num_train_steps=self.__num_train_steps
        )

    # @jit(target_backend='cuda')
    def __fit_model(self, model, tf_train_dataset, tf_validation_dataset):
        model_name = BERT_MODEL_NAME.split("/")[-1]
        # push_to_hub_model_id = f"{model_name}-finetuned-{self.__task}"

        tensorboard_callback = TensorBoard(log_dir="./text_classification_model_save/logs")

        # push_to_hub_callback = PushToHubCallback(
        #     output_dir="./text_classification_model_save",
        #     tokenizer=self.__tokenizer,
        #     hub_model_id=push_to_hub_model_id,
        #
        # )
        # metric_callback = KerasMetricCallback(
        #     metric_fn=compute_metrics, eval_dataset=tf_validation_dataset, label_cols=None,
        # )
        callbacks = [
            # metric_callback,
            tensorboard_callback,
            # push_to_hub_callback
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

    @staticmethod
    def create_classifier_model():
        id2label = {'0': "Rumor", '1': "Non Rumor"}
        label2id = {val: key for key, val in id2label.items()}

        model = TFAutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
        )
        return model
