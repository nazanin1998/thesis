import numpy as np
import pandas as pd
import tensorflow
from datasets import Dataset, DatasetDict, load_metric
from transformers import BertForSequenceClassification, BertForNextSentencePrediction, \
    AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, create_optimizer, AutoTokenizer
from transformers.keras_callbacks import PushToHubCallback
from keras.callbacks import TensorBoard
from evaluate import load

from lib.constants import PHEME_LABEL_COL_NAME
from lib.training_modules.bert.bert_configurations import BERT_EPOCHS, BERT_LEARNING_RATE, BERT_BATCH_SIZE


class BertNew:
    def __init__(self, train_df, val_df, test_df):
        self.__train_df = train_df
        self.__val_df = val_df
        self.__test_df = test_df
        self.GLUE_TASKS = [
            "cola",
            "mnli",
            "mnli-mm",
            "mrpc",
            "qnli",
            "qqp",
            "rte",
            "sst2",
            "stsb",
            "wnli",
        ]
        self.__task = "cola"
        self.__model_checkpoint = "distilbert-base-uncased"

        self.__train_ds = Dataset.from_pandas(train_df)
        self.__val_ds = Dataset.from_pandas(val_df)
        self.__test_ds = Dataset.from_pandas(test_df)

        self.__train_ds = self.__train_ds.rename_column(PHEME_LABEL_COL_NAME, 'label')
        self.__val_ds = self.__val_ds.rename_column(PHEME_LABEL_COL_NAME, 'label')
        self.__test_ds = self.__test_ds.rename_column(PHEME_LABEL_COL_NAME, 'label')

        # self.__train_ds = self.__train_ds.class_encode_column('label')
        # self.__val_ds = self.__val_ds.class_encode_column('label')
        # self.__test_ds = self.__test_ds.class_encode_column('label')

        self.__dataset = DatasetDict()

        self.__dataset['train'] = self.__train_ds
        self.__dataset['validation'] = self.__val_ds
        self.__dataset['test'] = self.__test_ds

        self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_checkpoint)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def start(self):
        self.do_preprocess()

    def do_preprocess(self):
        pre_tokenizer_columns = set(self.__dataset["train"].features)

        encoded_dataset = self.__dataset.map(self.preprocess_function, batched=True)
        tokenizer_columns = list(set(encoded_dataset["train"].features) - pre_tokenizer_columns)
        print("Columns added by tokenizer:", tokenizer_columns)
        print(f'labels: {encoded_dataset["train"].features["label"]}')

        id2label = {'0': "Rumor", '1': "Non Rumor"}
        label2id = {val: key for key, val in id2label.items()}

        model = TFAutoModelForSequenceClassification.from_pretrained(
            self.__model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
        )
        tf_train_dataset = model.prepare_tf_dataset(
            encoded_dataset["train"],
            shuffle=True,
            batch_size=BERT_BATCH_SIZE,
            tokenizer=self.__tokenizer
        )

        tf_validation_dataset = model.prepare_tf_dataset(
            encoded_dataset['validation'],
            shuffle=False,
            batch_size=BERT_BATCH_SIZE,
            tokenizer=self.__tokenizer,
        )

        print(f"tf_train_ds {tf_train_dataset}")
        print(f"tf_validation_dataset {tf_validation_dataset}")

        batches_per_epoch = len(encoded_dataset["train"]) // BERT_BATCH_SIZE
        total_train_steps = int(batches_per_epoch * BERT_EPOCHS)
        # actual_task = "mnli" if self.__task == "mnli-mm" else self.__task
        metric = load("glue", self.__task)
        input_spec, label_spec = tf_validation_dataset.element_spec
        print(f'input_spec {input_spec}')
        print(f'label_spec {label_spec}')
        input_spec, label_spec = tf_train_dataset.element_spec
        print(f'input_spec {input_spec}')

        from transformers.keras_callbacks import KerasMetricCallback
        # def compute_metrics(predictions, labels):
        #     decoded_predictions = self.__tokenizer.batch_decode(predictions, skip_special_tokens=True)
        #     decoded_labels = self.__tokenizer.batch_decode(labels, skip_special_tokens=True)
        #     result = metric.compute(predictions=decoded_predictions, references=decoded_labels)
        #     return {key: value.mid.fmeasure * 100 for key, value in result.items()}

        def compute_metrics(eval_predictions):

            predictions, labels = eval_predictions
            if self.__task != "stsb":
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = predictions[:, 0]
            print(f"baby label {labels}")
            return metric.compute(predictions=predictions, references=labels)

        model_name = self.__model_checkpoint.split("/")[-1]
        push_to_hub_model_id = f"{model_name}-finetuned-{self.__task}"

        tensorboard_callback = TensorBoard(log_dir="./text_classification_model_save/logs")

        # push_to_hub_callback = PushToHubCallback(
        #     output_dir="./text_classification_model_save",
        #     tokenizer=self.__tokenizer,
        #     hub_model_id=push_to_hub_model_id,
        #
        # )

        optimizer, schedule = create_optimizer(
            init_lr=BERT_LEARNING_RATE, num_warmup_steps=0, num_train_steps=total_train_steps
        )

        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # metrics = tensorflow.keras.metrics.SparseCategoricalAccuracy(
        #     'accuracy', dtype=tensorflow.float32)
        model.compile(optimizer=optimizer, loss=loss, )
        metric_callback = KerasMetricCallback(
            metric_fn=compute_metrics, eval_dataset=tf_validation_dataset, label_cols=None,
        )

        callbacks = [metric_callback, tensorboard_callback,
                     # push_to_hub_callback
                     ]

        model.fit(
            tf_train_dataset,
            validation_data=tf_validation_dataset,
            epochs=BERT_EPOCHS,
            callbacks=callbacks,
        )

        # print(f"baby {len(self.preprocess_function(self.__train_ds, tokenizer)['input_ids'])}")

    def preprocess_function(self, examples):
        return self.__tokenizer(examples['text'], examples['reaction_text'], list(map(str, examples['label'])),
                                padding='longest',
                                truncation=True)

    def do_bert(self):
        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification.
        )

        # Tell pytorch to run this model on the GPU.
        desc = model.cuda()
