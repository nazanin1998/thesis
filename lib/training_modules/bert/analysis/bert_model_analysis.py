import tensorflow as tf
from matplotlib import pyplot as plt

from lib.training_modules.bert.bert_configurations import BERT_EPOCHS
from lib.utils.log.logger import log_phase_desc

def compute_max_mean(items):
    mean_of = sum(items) / len(items)
    max_of = max(items)
    return max_of, mean_of

class BertModelAnalysis:
    def __init__(self, model, histories):
        self.__model = model
        self.__histories = histories

    def plot_bert_model(self):
        try:
            tf.keras.utils.plot_model(self.__model, to_file="abb.png")
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

    def evaluation(self, test_tensor_dataset= None):
        fold_index = 0
        
        train_acc_list = []
        validation_acc_list = []
        train_loss_list = []
        validation_loss_list = []
        
        for history in self.__histories:
            
            fold_index += 1
            print(f'FOLD {fold_index}')
            
            train_loss, validation_loss, train_acc, validation_acc = self.get_history_metrics(history)
            
            train_acc_list.extend(train_acc)
            train_loss_list.extend(train_loss)
            validation_acc_list.extend(validation_acc)
            validation_loss_list.extend(validation_loss)
            
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
     
        return train_acc_list, validation_acc_list, train_loss_list, validation_loss_list, validation_acc_mean, validation_loss_mean, validation_acc_max, validation_loss_max

    def get_history_metrics(self):
        history_dict = self.__history.history

        train_loss = history_dict['loss']
        validation_loss = history_dict['val_loss']
        train_acc = history_dict['accuracy']
        validation_acc = history_dict['val_accuracy']

        return train_loss, validation_loss, train_acc, validation_acc
