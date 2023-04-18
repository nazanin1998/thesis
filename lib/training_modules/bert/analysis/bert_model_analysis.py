from lib.models.evaluation_model import EvaluationModel
from lib.models.metrics_model import MetricsModel, compute_max_mean
import tensorflow as tf
from matplotlib import pyplot as plt

from lib.training_modules.bert.bert_configurations import BERT_EPOCHS, BERT_EPOCHS_K_FOLD, BERT_K_FOLD, BERT_USE_K_FOLD
from lib.utils.log.logger import log_phase_desc
from tabulate import tabulate

import math


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
    def plot_bert_evaluation_metrics(eval_res):
        eval_result = EvaluationModel(
            train=MetricsModel(
                accuracy=[1, 1, 0.5, 0.9, 0.8 ,0.8],
                f1_score=[1, 1, 0.5, 0.9, 0.8 ,0.8],
                loss=[1, 1, 0.5, 0.9, 0.8 ,0.8], 
                precision=[1, 1, 0.5, 0.9, 0.8 ,0.8], 
                recall=[1, 1, 0.5, 0.9, 0.8 ,0.8]),
            validation=MetricsModel(
                accuracy=[1, 1, 0.5, 0.9, 0.8 ,0.8],
                f1_score=[1, 1, 0.5, 0.9, 0.8 ,0.8],
                loss=[1, 1, 0.5, 0.9, 0.8 ,0.8], 
                precision=[1, 1, 0.5, 0.9, 0.8 ,0.8], 
                recall=[1, 1, 0.5, 0.9, 0.8 ,0.8]),
            test=MetricsModel(accuracy=1,f1_score=1,loss=1, precision=1, recall=1)
        )
        fig = plt.figure(figsize=(2, 5))
        fig.tight_layout()
        
        x_points = []
        for i in range (1, eval_result.get_epoch_len()+1):
           x_points.append(i)
            

        plt.subplot(5, 1, 1)
        # r is for "solid red line"
        # b is for "solid blue line"
        plt.plot(x_points, eval_result.get_train().get_loss(), 'r', label='Training loss')
        plt.plot(x_points, eval_result.get_validation().get_loss(), 'b', label='Validation loss')
        plt.title('Training Loss vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(5, 1, 2)
        plt.plot(x_points, eval_result.get_train().get_accuracy(), 'r', label='Training acc')
        plt.plot(x_points, eval_result.get_validation().get_accuracy(), 'b', label='Validation acc')
        plt.title('Training Accuracy vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        # plt.legend(loc='lower right')


        plt.subplot(5, 1, 3)
        plt.plot(x_points,  eval_result.get_train().get_recall(), 'r', label='Training Recall')
        plt.plot(x_points,  eval_result.get_validation().get_recall(), 'b', label='Validation Recall')
        plt.title('Training Recall vs Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        # plt.legend(loc='lower right')
        plt.savefig("plot_bert.png")
        
        plt.subplot(5, 1, 4)
        plt.plot(x_points,  eval_result.get_train().get_precision(), 'r', label='Training Precision')
        plt.plot(x_points,  eval_result.get_validation().get_precision(), 'b', label='Validation Precision')
        plt.title('Training Precision vs Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        # plt.legend(loc='lower right')
        
        plt.subplot(5, 1, 5)
        plt.plot(x_points,  eval_result.get_train().get_f1_score(), 'r', label='Training F1 Score')
        plt.plot(x_points,  eval_result.get_train().get_f1_score(), 'b', label='Validation F1 Score')
        plt.title('Training F1 Score vs Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
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
            

        test_metrics=''
        if not BERT_USE_K_FOLD:
            result = self.__model.evaluate(test_tensor_dataset)
            test_metrics = MetricsModel(
                accuracy= result[1], 
                precision=result[2], 
                recall=result[3], 
                loss= result[0], 
                f1_score=result[4],
                )
        
        train_total_metrics = MetricsModel(
            accuracy= train_acc_list, 
            precision=train_precision_list, 
            recall=train_recall_list, 
            loss= train_loss_list, 
            f1_score=train_f1_score_list)
        
        val_total_metrics = MetricsModel(
            accuracy= validation_acc_list, 
            precision=validation_precision_list, 
            recall=validation_recall_list, 
            loss= validation_loss_list, 
            f1_score=validation_f1_score_list)
        
                
        eval_res = EvaluationModel(train=train_total_metrics, validation=val_total_metrics, test= test_metrics)
        self.print_evaluation_result(eval_res)
        return eval_res


    def print_evaluation_result(self, eval_result):
        # eval_result = EvaluationModel(
        #     train=MetricsModel(
        #         accuracy=[1, 1, 0.5, 0.9, 0.8 ,0.8],
        #         f1_score=[1, 1, 0.5, 0.9, 0.8 ,0.8],
        #         loss=[1, 1, 0.5, 0.9, 0.8 ,0.8], 
        #         precision=[1, 1, 0.5, 0.9, 0.8 ,0.8], 
        #         recall=[1, 1, 0.5, 0.9, 0.8 ,0.8]),
        #     validation=MetricsModel(
        #         accuracy=[1, 1, 0.5, 0.9, 0.8 ,0.8],
        #         f1_score=[1, 1, 0.5, 0.9, 0.8 ,0.8],
        #         loss=[1, 1, 0.5, 0.9, 0.8 ,0.8], 
        #         precision=[1, 1, 0.5, 0.9, 0.8 ,0.8], 
        #         recall=[1, 1, 0.5, 0.9, 0.8 ,0.8]),
        #     test=MetricsModel(accuracy=1,f1_score=1,loss=1, precision=1, recall=1)
        # )
        data = eval_result.to_table_array()
            
        headers = ['Metric Name']
        if BERT_USE_K_FOLD:
            for i in range (1 , eval_result.get_epoch_len()+1):
                epoch_num = i % BERT_EPOCHS_K_FOLD
                if epoch_num == 0:
                    epoch_num = BERT_EPOCHS_K_FOLD
                fold_num = math.ceil(i / BERT_EPOCHS_K_FOLD)
                headers.append(f"Fold-{fold_num}/Epoch-{epoch_num}")
        else:
            for i in range (1 , eval_result.get_epoch_len()+1):
                headers.append(f"Epoch-{i}")
            
        headers.append("Max")
        headers.append("Mean")
        
        table = tabulate(data, headers=headers, tablefmt='orgtbl')

        if not BERT_USE_K_FOLD:
            test_res = eval_result.get_test()
            print(f"TEST RESULT => Accuracy:  {test_res.get_accuracy()}, ")
            print(f"               Recall:    {test_res.get_recall()}, ")
            print(f"               Precision: {test_res.get_precision()}, ")
            print(f"               F1 Score:  {test_res.get_f1_score()}, ")
            print(f"               Loss:      {test_res.get_loss()}, ")
            
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
