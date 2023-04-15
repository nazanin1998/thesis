import tensorflow as tf
from matplotlib import pyplot as plt

from lib.training_modules.base.analysis.base_analysis import get_history_metrics
from lib.training_modules.bert.bert_configurations import BERT_EPOCHS
from lib.utils.log.logger import log_phase_desc


def compute_max_mean(items):
    mean_of = sum(items) / len(items)
    max_of = max(items)
    return max_of, mean_of


class BertModelAnalysis:
    def __init__(self, model, history):
        self.__model = model
        self.__history = history

    def plot_bert_model(self):
        try:
            tf.keras.utils.plot_model(self.__model, to_file="bert_model.png")
        except:
            print()

    @staticmethod
    def plot_bert_evaluation_metrics(train_acc, val_acc, train_loss, val_loss):
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()
        plt.subplot(2, 1, 1)
        # r is for "solid red line"
        # b is for "solid blue line"
        plt.plot(BERT_EPOCHS, train_loss, 'r', label='Training loss')
        plt.plot(BERT_EPOCHS, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(BERT_EPOCHS, train_acc, 'r', label='Training acc')
        plt.plot(BERT_EPOCHS, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig("plot_bert.png")

    def evaluation(self, test_tensor_dataset=None):
        if isinstance(self.__history, list):
            fold_index = 0
            train_acc_list = []
            validation_acc_list = []
            train_loss_list = []
            validation_loss_list = []
            for hist in self.__history:
                fold_index += 1
                print(f'FOLD {fold_index}')
                train_loss, validation_loss, train_acc, validation_acc = get_history_metrics(hist)
                train_loss_list.extend(train_loss)
                validation_loss_list.extend(validation_loss)
                train_acc_list.extend(train_acc)
                validation_acc_list.extend(validation_acc)
            train_acc_max, train_acc_mean = compute_max_mean(train_acc_list)
            train_loss_max, train_loss_mean = compute_max_mean(train_loss_list)
            validation_acc_max, validation_acc_mean = compute_max_mean(validation_acc_list)
            validation_loss_max, validation_loss_mean = compute_max_mean(validation_loss_list)

            log_phase_desc(f'Train       => Accuracy: {train_acc_list}, Loss: {train_loss_list}')
            log_phase_desc(f'Test        => Accuracy: {validation_acc_list}, Loss: {validation_loss_list}\n')
            log_phase_desc(f'Train(MEAN) => Accuracy: {train_acc_mean}, Loss: {train_loss_mean}')
            log_phase_desc(f'Train(MAX) => Accuracy: {train_acc_max}, Loss: {train_loss_max}\n')
            log_phase_desc(f'Test (MEAN) => Accuracy: {validation_acc_mean}, Loss: {validation_loss_mean}')
            log_phase_desc(f'Test (MAX) => Accuracy: {validation_acc_max}, Loss: {validation_loss_max}')
            return train_acc_list, validation_acc_list, train_loss_list, validation_loss_list

        else:

            train_loss, validation_loss, train_acc, validation_acc = get_history_metrics(self.__history)
            test_loss, test_accuracy = self.__model.evaluate(test_tensor_dataset)

            log_phase_desc(f'Training   => Accuracy: {train_acc}, Loss: {train_loss}')
            log_phase_desc(f'Validation => Accuracy: {validation_acc}, Loss: {validation_loss}')
            log_phase_desc(f'Test       => Accuracy: {test_accuracy}, Loss: {test_loss}')

            return train_acc, validation_acc, train_loss, validation_loss, test_loss, test_accuracy