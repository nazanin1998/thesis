from lib.utils.constants import TRAIN


class EvaluationModel:
    
    def __init__(self, train, validation, test):
        self.__train = train
        self.__validation = validation
        self.__test = test
        
    def get_train(self):
        return self.__train
    
    def get_test(self):
        return self.__test
    
    def get_validation(self):
        return self.__validation

    def to_table_array(self):
        return [
            self.__train.acc_to_table(title_prefix='TRAIN-'), 
            self.__train.recall_to_table(title_prefix='TRAIN-'), 
            self.__train.precision_to_table(title_prefix='TRAIN-'), 
            self.__train.f1_to_table(title_prefix='TRAIN-'), 
            self.__train.loss_to_table(title_prefix='TRAIN-'), 
            
            self.__validation.acc_to_table(title_prefix='VALIDATION-'), 
            self.__validation.recall_to_table(title_prefix='VALIDATION-'), 
            self.__validation.precision_to_table(title_prefix='VALIDATION-'), 
            self.__validation.f1_to_table(title_prefix='VALIDATION-'), 
            self.__validation.loss_to_table(title_prefix='VALIDATION-'), 
        ]

    def get_epoch_len(self):
        return len(self.__train.get_accuracy())