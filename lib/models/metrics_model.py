from lib.training_modules.bert.analysis.bert_model_analysis import compute_max_mean


class MetricsModel:
    def __init__(self, accuracy, precision, recall, f1_score, loss):
        self.__accuracy = accuracy
        self.__precision = precision
        self.__recall = recall
        self.__f1_score = f1_score
        self.__loss = loss

    def get_accuracy(self):
        return self.__accuracy
    
    def get_recall(self):
        return self.__recall
    
    def get_precision(self):
        return self.__precision
    
    def get_f1_score(self):
        return self.__f1_score
    
    def get_loss(self):
        return self.__loss

    def __to_table(self, items, title):
        l = []
        max_of, mean_of = compute_max_mean(items)
       
        l.append(title)
        l.extend(items)
        l.append(max_of)
        l.append(mean_of)
        
        return l
    
    def acc_to_table(self, title_prefix=''):
        return self.__to_table(self.__accuracy, f'{title_prefix}Accuracy')   
    
    def recall_to_table(self, title_prefix=''):
        return self.__to_table(self.__recall, f'{title_prefix}Recall')
    
    def precision_to_table(self, title_prefix=''):
        return self.__to_table(self.__precision, f'{title_prefix}Precision')
    
    def f1_to_table(self, title_prefix=''):
        return self.__to_table(self.__f1_score, f'{title_prefix}F1 Score')
    
    def loss_to_table(self, title_prefix=''):
        return self.__to_table(self.__accuracy, f'{title_prefix}Loss')
    
    