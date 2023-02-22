import tensorflow as tf
from matplotlib import pyplot as plt

from lib.training_modules.bert.bert_configurations import BERT_EPOCHS
from lib.utils.log.logger import log_phase_desc


class BertModelAnalysis:
    def __init__(self, model, history):
        self.__model = model
        self.__history = history

    def plot_bert_model(self):
        try:
            tf.keras.utils.plot_model(self.__model, to_file="abb.png")
        except:
            print()

    @staticmethod
    def plot_bert_evaluation_metrics(acc, val_acc, loss, val_loss):
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        # r is for "solid red line"
        # b is for "solid blue line"
        plt.plot(BERT_EPOCHS, loss, 'r', label='Training loss')
        plt.plot(BERT_EPOCHS, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(BERT_EPOCHS, acc, 'r', label='Training acc')
        plt.plot(BERT_EPOCHS, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig("plot_bert.png")

    def evaluation(self, test_tensor_dataset):

        train_loss, validation_loss, train_acc, validation_acc = self.get_history_metrics()
        test_loss, test_accuracy = self.__model.evaluate(test_tensor_dataset)

        log_phase_desc(f'Training   => Accuracy: {train_acc}, Loss: {train_loss}')
        log_phase_desc(f'Validation => Accuracy: {validation_acc}, Loss: {validation_loss}')
        log_phase_desc(f'Test       => Accuracy: {test_accuracy}, Loss: {test_loss}')

        return train_acc, validation_acc, train_loss, validation_loss, test_loss, test_accuracy

    def get_history_metrics(self):
        history_dict = self.__history.history

        train_loss = history_dict['loss']
        validation_loss = history_dict['val_loss']
        train_acc = history_dict['accuracy']
        validation_acc = history_dict['val_accuracy']

        return train_loss, validation_loss, train_acc, validation_acc
