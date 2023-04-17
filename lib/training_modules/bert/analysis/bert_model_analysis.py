from lib.models.evaluation_model import EvaluationModel
from lib.models.metrics_model import MetricsModel, compute_max_mean
import tensorflow as tf
from matplotlib import pyplot as plt

from lib.training_modules.bert.bert_configurations import BERT_EPOCHS, BERT_EPOCHS_K_FOLD, BERT_K_FOLD, BERT_USE_K_FOLD
from lib.utils.log.logger import log_phase_desc
from tabulate import tabulate



class BertModelAnalysis:
    def __init__(self, model, histories):
        self.__model = model
        self.__histories = histories

    def plot_bert_model(self):
        try:
            tf.keras.utils.plot_model(self.__model, to_file="bert_model.png")
        except:
            print()

    @staticmethod
    def plot_bert_evaluation_metrics(train_acc, val_acc, train_loss, val_loss):
        
        fig = plt.figure(figsize=(10, 10))
        fig.tight_layout()
        
        x_points = []
        for i in range (1, len(train_acc)+1):
           x_points.append(i)
            

        plt.subplot(2, 1, 1)
        # r is for "solid red line"
        # b is for "solid blue line"
        plt.plot(x_points, train_loss, 'r', label='Training loss')
        plt.plot(x_points, val_loss, 'b', label='Validation loss')
        plt.title('Training Loss vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(x_points, train_acc, 'r', label='Training acc')
        plt.plot(x_points, val_acc, 'b', label='Validation acc')
        plt.title('Training Accuracy vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig("plot_bert.png")

    def evaluation(self, test_tensor_dataset = None):
        fold_index = 0
        
        train_acc_list = []
        train_recall_list = []
        train_precision_list = []
        train_f1_score_list = []
        train_loss_list = []
        
        validation_acc_list = []
        validation_recall_list = []
        validation_precision_list = []
        validation_f1_score_list = []
        validation_loss_list = []
        
        for history in self.__histories:
            
            fold_index += 1
            
            train_metrics, val_metrics = self.get_history_metrics(history)

            train_loss_list.extend(train_metrics.get_loss())
            train_acc_list.extend(train_metrics.get_accuracy())
            train_recall_list.extend(train_metrics.get_recall())
            train_f1_score_list.extend(train_metrics.get_f1_score())
            train_precision_list.extend(train_metrics.get_precision())
            
            validation_loss_list.extend(val_metrics.get_loss())
            validation_acc_list.extend(val_metrics.get_accuracy())
            validation_recall_list.extend(val_metrics.get_recall())
            validation_f1_score_list.extend(val_metrics.get_f1_score())
            validation_precision_list.extend(val_metrics.get_precision())
            
        train_acc_max, train_acc_mean = compute_max_mean(train_acc_list)
        train_loss_max, train_loss_mean = compute_max_mean(train_loss_list)
        train_recall_max, train_recall_mean = compute_max_mean(train_recall_list)
        train_f1_score_max, train_f1_score_mean = compute_max_mean(train_f1_score_list)
        train_precision_max, train_precision_mean = compute_max_mean(train_precision_list)
        
        validation_acc_max, validation_acc_mean = compute_max_mean(validation_acc_list)
        validation_loss_max, validation_loss_mean = compute_max_mean(validation_loss_list)
        validation_recall_max, validation_recall_mean = compute_max_mean(validation_recall_list)
        validation_f1_score_max, validation_f1_score_mean = compute_max_mean(validation_f1_score_list)
        validation_precision_max, validation_precision_mean = compute_max_mean(validation_precision_list)

        log_phase_desc(f'Train             => Accuracy: {train_acc_list}, Loss: {train_loss_list}')
        log_phase_desc(f'Validation        => Accuracy: {validation_acc_list}, Loss: {validation_loss_list}\n')
        log_phase_desc(f'Train(MEAN)       => Accuracy: {train_acc_mean}, Loss: {train_loss_mean}')
        log_phase_desc(f'Train(MAX)        => Accuracy: {train_acc_max}, Loss: {train_loss_max}\n')
        log_phase_desc(f'Validation (MEAN) => Accuracy: {validation_acc_mean}, Loss: {validation_loss_mean}')
        log_phase_desc(f'Validation (MAX)  => Accuracy: {validation_acc_max}, Loss: {validation_loss_max}')
        
        test_loss, test_accuracy =0,0
        test_metrics=''
        if not BERT_USE_K_FOLD:
            print(self.__model.metrics_names)
            # ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
            result = self.__model.evaluate(test_tensor_dataset)
            test_metrics = MetricsModel(
                accuracy= result[1], 
                precision=result[2], 
                recall=result[3], 
                loss= result[0], 
                f1_score=result[4],
                )
            log_phase_desc(f'Test              => {result}')
        
        train_total_metrics = MetricsModel(
            accuracy= train_acc_list, 
            precision=train_precision_list, 
            recall=train_recall_list, 
            loss= train_loss_list, 
            f1_score=train_f1_score_list,)
        
        val_total_metrics = MetricsModel(
            accuracy= validation_acc_list, 
            precision=validation_precision_list, 
            recall=validation_recall_list, 
            loss= validation_loss_list, 
            f1_score=validation_f1_score_list,)
        
                
        eval_res = EvaluationModel(train=train_total_metrics, validation=val_total_metrics, test= test_metrics)
        print(eval_res)
        self.print_evaluation_result(eval_res)
        return train_acc_list, validation_acc_list, train_loss_list, validation_loss_list, validation_acc_mean, validation_loss_mean, validation_acc_max, validation_loss_max, test_loss, test_accuracy


    def print_evaluation_result(self, eval_result):
        data = eval_result.to_table_array()
            
        headers = ['Metric Name']
        if BERT_USE_K_FOLD:
            idx = 0
            for res in eval_result:
                idx +=1
                epoch_num = idx % BERT_EPOCHS_K_FOLD
                fold_num = round(idx / BERT_EPOCHS_K_FOLD)
                headers.append(f"Fold-{fold_num}/Epoch-{epoch_num}")
        else:
            for i in range (1 , eval_result.get_epoch_len()+1):
                headers.append(f"Epoch-{i}")
            
        headers.append("Max")
        headers.append("Mean")
        
        table = tabulate(data, headers=headers, tablefmt='orgtbl')

        print(table)

    def get_history_metrics(self, hist):
        history_dict = hist.history

        train_metrics = MetricsModel(
            accuracy= history_dict['accuracy'], 
            precision=history_dict['precision'], 
            recall=history_dict['recall'], 
            loss= history_dict['loss'], 
            f1_score=history_dict['f1_score'], 
        )
       
        val_metrics = MetricsModel(
            accuracy= history_dict['val_accuracy'], 
            precision=history_dict['val_precision'], 
            recall=history_dict['val_recall'], 
            loss= history_dict['val_loss'], 
            f1_score=history_dict['val_f1_score'], 
        )
       
        return train_metrics, val_metrics
