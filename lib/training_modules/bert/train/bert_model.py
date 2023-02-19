from abc import ABC


class MyBertModel(ABC):
    def build_classifier_model(self, num_classes):
        pass

    def start(self,
              label_classes,
              x_train_tensor,
              x_val_tensor,
              x_test_tensor,
              y_train_tensor,
              y_val_tensor,
              y_test_tensor,
              bert_preprocess_model):
        pass

    def build_classifier_model(self_out, num_classes):
        pass

    def plot_model(self, acc, val_acc, loss, val_loss):
        pass

    def evaluation(self,
                   classifier_model,
                   test_tensor_dataset):
        pass

    def save_model(self,
                   bert_preprocess_model,
                   classifier_model,
                   save_path='./saved_models',
                   ):
        pass

    def get_metrics_and_loss(self):
        pass
