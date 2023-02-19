import os

from matplotlib import pyplot as plt
from transformers import AdamW

from lib.training_modules.bert.bert_configurations import bert_dropout_rate, bert_batch_size, \
    bert_epochs
from lib.training_modules.bert.bert_model_name import get_bert_model_name
from lib.training_modules.bert.train.bert_model import MyBertModel
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text  # A dependency of the preprocessing model
import tensorflow_addons as tfa

from lib.utils.log.logger import log_start_phase, log_end_phase, log_line, print_indented_key_value, print_indented, \
    log_phase_desc


class BertModelImpl(MyBertModel):
    def __init__(self,
                 bert_model_name='bert_en_uncased_L-12_H-768_A-12'):
        self.bert_model_name = get_bert_model_name(bert_model_name=bert_model_name)

    def start(self,
              train_tensor_dataset,
              val_tensor_dataset,
              test_tensor_dataset,
              label_classes,
              train_len,
              validation_len,
              test_len,
              bert_preprocess_model,
              ):
        log_start_phase(3, 'BERT ON TWEET TEXT')
        log_phase_desc(f'Bert Model: {self.bert_model_name}')

        metrics, loss = self.get_metrics_and_loss()

        steps_per_epoch = train_len // bert_batch_size
        validation_steps = validation_len // bert_batch_size
        num_train_steps = steps_per_epoch * bert_epochs
        num_warmup_steps = num_train_steps // 10

        classifier_model = self.build_classifier_model(label_classes)

        # optimizer = optimization.create_optimizer(
        #     init_lr=init_lr,
        #     num_train_steps=num_train_steps,
        #     num_warmup_steps=num_warmup_steps,
        #     optimizer_type='adamw')

        print("adam")
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

        classifier_model.compile(
            optimizer=optimizer,
            loss=loss, metrics=[metrics])
        print("optimizer")

        classifier_model.fit(
            x=train_tensor_dataset,
            batch_size=bert_batch_size,
            validation_data=val_tensor_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=bert_epochs,
            validation_steps=validation_steps,
        )
        print("fit")

        try:
            classifier_model.summary()
            tf.keras.utils.plot_model(classifier_model)
        except:
            print()

        self.save_model(
            bert_preprocess_model=bert_preprocess_model,
            classifier_model=classifier_model)

        acc, val_acc, loss, val_loss = self.evaluation(
            classifier_model=classifier_model,
            test_tensor_dataset=test_tensor_dataset)
        # classifier_model.prz
        self.plot_model(
            acc=acc,
            val_acc=val_acc,
            loss=loss,
            val_loss=val_loss)

        log_end_phase(3, 'BERT ON TWEET TEXT')
        log_line()

    def plot_model(self, acc, val_acc, loss, val_loss):
        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        # r is for "solid red line"
        plt.plot(epochs, loss, 'r', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

    def evaluation(self,
                   classifier_model,
                   test_tensor_dataset):
        first_loss, accuracy = classifier_model.evaluate(test_tensor_dataset, )

        history_dict = classifier_model.history
        acc, val_acc, loss, val_loss = 0, 0, 0, 0
        try:
            acc = history_dict['binary_accuracy']
            log_phase_desc(f'Accuracy: {acc}')
        except:
            print()
        try:
            val_acc = history_dict['val_binary_accuracy']
            log_phase_desc(f'Val Acc : {val_acc}')
        except:
            print()
        try:
            loss = history_dict['loss']
            log_phase_desc(f'Loss    : {loss}')
        except:
            print()
        try:
            val_loss = history_dict['val_loss']
            log_phase_desc(f'Val Loss: {val_loss}')
        except:
            print()

        log_phase_desc(f'Loss    : {first_loss}')
        log_phase_desc(f'Accuracy: {accuracy}')

        return acc, val_acc, loss, val_loss

    def save_model(self,
                   bert_preprocess_model,
                   classifier_model,
                   save_path='./saved_models',
                   ):

        bert_type_name = self.bert_model_name.split('/')[-2]

        saved_model_name = f'pheme_{bert_type_name}'

        saved_model_path = os.path.join(save_path, saved_model_name)

        preprocess_inputs = bert_preprocess_model.inputs
        bert_encoder_inputs = bert_preprocess_model(preprocess_inputs)
        bert_outputs = classifier_model(bert_encoder_inputs)
        model_for_export = tf.keras.Model(preprocess_inputs, bert_outputs)

        # Save everything on the Colab host (even the variables from TPU memory)
        # save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
        model_for_export.save(saved_model_path, include_optimizer=False,
                              # options=save_options
                              )
        print(f'SAVE {saved_model_name} model (PATH): {saved_model_path}')

    def get_metrics_and_loss(self):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)
        log_phase_desc(f'Metrics: {metrics}')
        log_phase_desc(f'Loss: {loss}')

        return metrics, loss

    def build_classifier_model(self_out, num_classes):
        class Classifier(tf.keras.Model):
            def __init__(self, num_classes):
                super(Classifier, self).__init__(name="prediction")
                self.encoder = hub.KerasLayer(self_out.bert_model_name, trainable=True)
                self.dropout = tf.keras.layers.Dropout(bert_dropout_rate)
                self.dense = tf.keras.layers.Dense(num_classes)

            def call(self, preprocessed_text):
                encoder_outputs = self.encoder(preprocessed_text)
                pooled_output = encoder_outputs["pooled_output"]
                x = self.dropout(pooled_output)
                x = self.dense(x)
                return x

        model = Classifier(num_classes)

        return model
