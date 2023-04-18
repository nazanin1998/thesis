from lib.models.evaluation_model import EvaluationModel
from lib.models.metrics_model import MetricsModel
from matplotlib import pyplot as plt
from lib.training_modules.base.analysis.base_analysis import convert_test_eval_result_to_metric_model, get_history_metrics, get_moch_evaluatio_data, make_header_for_eval_table, plot_model

from lib.training_modules.bert.bert_configurations import BERT_EPOCHS, BERT_EPOCHS_K_FOLD, BERT_K_FOLD, BERT_USE_K_FOLD
from lib.utils.log.logger import log_phase_desc
from tabulate import tabulate



class BertModelAnalysis:
    def __init__(self, model, histories):
        self.__model = model
        self.__histories = histories

    def plot_bert_model(self):
        plot_model(self.__model)
    
    
    @staticmethod
    def plot_bert_evaluation_metrics(eval_result):
        fig = plt.figure(figsize=(20, 20))
        fig.tight_layout()
        
        x_points = []
        for i in range (1, eval_result.get_epoch_len()+1):
           x_points.append(i)
            

        plt.subplot(321)
        plt.plot(x_points, eval_result.get_train().get_loss(), 'r', label='Training loss')
        plt.plot(x_points, eval_result.get_validation().get_loss(), 'b', label='Validation loss')
        plt.title('Training Loss vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(322)
        plt.plot(x_points, eval_result.get_train().get_accuracy(), 'r', label='Training acc')
        plt.plot(x_points, eval_result.get_validation().get_accuracy(), 'b', label='Validation acc')
        plt.title('Training Accuracy vs Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)


        plt.subplot(323)
        plt.plot(x_points,  eval_result.get_train().get_recall(), 'r', label='Training Recall')
        plt.plot(x_points,  eval_result.get_validation().get_recall(), 'b', label='Validation Recall')
        plt.title('Training Recall vs Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.grid(True)
        
        plt.subplot(324)
        plt.plot(x_points,  eval_result.get_train().get_precision(), 'r', label='Training Precision')
        plt.plot(x_points,  eval_result.get_validation().get_precision(), 'b', label='Validation Precision')
        plt.title('Training Precision vs Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.grid(True)
        
        plt.subplot(325)
        plt.plot(x_points,  eval_result.get_train().get_f1_score(), 'r', label='Training F1 Score')
        plt.plot(x_points,  eval_result.get_train().get_f1_score(), 'b', label='Validation F1 Score')
        plt.title('Training F1 Score vs Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True)
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
            
            train_metrics, val_metrics = get_history_metrics(history)

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
        
        if not BERT_USE_K_FOLD:
            result = self.__model.evaluate(test_tensor_dataset)
            test_metrics = convert_test_eval_result_to_metric_model(result)    
            eval_res = EvaluationModel(train=train_total_metrics, validation=val_total_metrics, test= test_metrics)
        else:
            eval_res = EvaluationModel(train=train_total_metrics, validation=val_total_metrics)
        
        self.print_evaluation_result(eval_res)
        return eval_res


    def print_evaluation_result(self, eval_result):
        data = eval_result.to_table_array()
            
        headers = make_header_for_eval_table(eval_result)
        
        table = tabulate(data, headers=headers, tablefmt='orgtbl')

        if not BERT_USE_K_FOLD:
            test_res = eval_result.get_test()
            print(f"TEST RESULT => Accuracy:  {test_res.get_accuracy()}, ")
            print(f"               Recall:    {test_res.get_recall()}, ")
            print(f"               Precision: {test_res.get_precision()}, ")
            print(f"               F1 Score:  {test_res.get_f1_score()}, ")
            print(f"               Loss:      {test_res.get_loss()}, ")
            
        print(table)

   