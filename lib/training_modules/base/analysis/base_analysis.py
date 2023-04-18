from lib.training_modules.bert.bert_configurations import BERT_EPOCHS_K_FOLD, BERT_USE_K_FOLD
import tensorflow as tf
from lib.models.evaluation_model import EvaluationModel
from lib.models.metrics_model import MetricsModel
import math

def plot_model(model, save_file_name = "bert_model.png"):
    try:
        tf.keras.utils.plot_model(model, to_file=save_file_name)
    except:
        print()
        
def get_moch_evaluatio_data():
    return EvaluationModel(
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
def get_history_metrics(hist):
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

def convert_test_eval_result_to_metric_model(result):
    return MetricsModel(
        accuracy= result[1], 
        precision=result[2], 
        recall=result[3], 
        loss= result[0], 
        f1_score=result[4],
    )
    
def make_header_for_eval_table(eval_result):
    headers = ['Metric Name']
    header_len_plus_one = eval_result.get_epoch_len() +1
    
    if BERT_USE_K_FOLD:
        
        for i in range (1 , header_len_plus_one):
            fold_num = math.ceil(i / BERT_EPOCHS_K_FOLD)
            
            epoch_num = i % BERT_EPOCHS_K_FOLD
            if epoch_num == 0:
                epoch_num = BERT_EPOCHS_K_FOLD
           
            headers.append(f"Fold-{fold_num}/Epoch-{epoch_num}")
    else:
       
        for i in range (1 , eval_result.get_epoch_len()+1):
            headers.append(f"Epoch-{i}")
        
    headers.append("Max")
    headers.append("Mean")
    return headers