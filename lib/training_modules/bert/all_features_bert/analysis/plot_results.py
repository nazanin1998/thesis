import numpy as np
from matplotlib import pyplot as plt

from lib.models.evaluation_model import EvaluationModel
from lib.models.metrics_model import MetricsModel


def create_result(
        train_acc_list,
        train_precision_list,
        train_recall_list,
        train_loss_list,
        train_f1_score_list,
        validation_acc_list,
        validation_precision_list,
        validation_recall_list,
        validation_loss_list,
        validation_f1_score_list,
):
    print('b')
    train_total_metrics = MetricsModel(
        accuracy=train_acc_list,
        precision=train_precision_list,
        recall=train_recall_list,
        loss=train_loss_list,
        f1_score=train_f1_score_list)

    val_total_metrics = MetricsModel(
        accuracy=validation_acc_list,
        precision=validation_precision_list,
        recall=validation_recall_list,
        loss=validation_loss_list,
        f1_score=validation_f1_score_list)

    # if not BERT_USE_K_FOLD:
    #     result = self.__model.evaluate(test_tensor_dataset)
    #     test_metrics = convert_test_eval_result_to_metric_model(result)
    eval_res = EvaluationModel(train=train_total_metrics, validation=val_total_metrics, )
    # else:
    #     eval_res = EvaluationModel(train=train_total_metrics, validation=val_total_metrics)
    return eval_res


def plot_result(eval_result, res_name):
    fig = plt.figure(figsize=(30, 20))
    # fig = plt.figure(figsize=(20, 20)) // if all metrics plot
    fig.tight_layout()

    x_points = []
    for i in range(1, eval_result.get_epoch_len() + 1):
        x_points.append(i)

    plt.subplot(221)  # 321
    plt.plot(x_points, eval_result.get_train().get_loss(), 'r', label='Training Loss')
    plt.plot(x_points, eval_result.get_validation().get_loss(), 'b', label='Validation Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.xticks(np.arange(min(x_points), max(x_points) + 1, 1.0))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(222)  # 322
    plt.plot(x_points, eval_result.get_train().get_accuracy(), 'r', label='Training Accuracy')
    plt.plot(x_points, eval_result.get_validation().get_accuracy(), 'b', label='Validation Accuracy')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xticks(np.arange(min(x_points), max(x_points) + 1, 1.0))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(223)  # 323
    plt.plot(x_points, eval_result.get_train().get_recall(), 'r', label='Training Recall')
    plt.plot(x_points, eval_result.get_validation().get_recall(), 'b', label='Validation Recall')
    plt.title('Training Recall vs Validation Recall')
    plt.xticks(np.arange(min(x_points), max(x_points) + 1, 1.0))
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True)

    if eval_result.get_train().get_precision() is not None:
        plt.subplot(224)  # 324
        plt.plot(x_points, eval_result.get_train().get_precision(), 'r', label='Training Precision')
        plt.plot(x_points, eval_result.get_validation().get_precision(), 'b', label='Validation Precision')
        plt.xticks(np.arange(min(x_points), max(x_points) + 1, 1.0))
        plt.title('Training Precision vs Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.grid(True)

    if eval_result.get_train().get_precision() is not None:
        plt.subplot(325)  # 325
        plt.plot(x_points, eval_result.get_train().get_f1_score(), 'r', label='Training F1 Score')
        plt.plot(x_points, eval_result.get_validation().get_f1_score(), 'b', label='Validation F1 Score')
        plt.title('Training F1 Score vs Validation F1 Score')
        plt.xticks(np.arange(min(x_points), max(x_points) + 1, 1.0))
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True)
    plt.savefig(res_name + ".png")


eval_res1 = create_result(
    train_acc_list=[0.756005, 0.868995, 0.923265, 0.956628, 0.975311],
    train_loss_list=[0.4705, 0.2936, 0.1913, 0.1100, 0.0707],
    train_recall_list=[0.548043, 0.841415, 0.915036, 0.955294, 0.975311],
    train_precision_list=None,
    train_f1_score_list=None,
    validation_acc_list=[0.8375, 0.830208, 0.845833, 0.866667, 0.860417],
    validation_loss_list=[0.3888, 0.3649, 0.3924, 0.4460, 0.4551],
    validation_recall_list=[0.759375, 0.833333, 0.903125, 0.93125, 0.944792],
    validation_precision_list=None,
    validation_f1_score_list=None
)

eval_res4 = create_result(
    train_acc_list=[
        0.671708, 0.843639, 0.903247, 0.94395, 0.968861, 0.97976, 0.987767, 0.988879, 0.993105, 0.993772],
    train_loss_list=[0.6004, 0.3633, 0.2477, 0.1558, 0.0876, 0.0654, 0.0385, 0.0408, 0.0247, 0.0193, ],
    train_recall_list=[
        0.289813, 0.795819, 0.808274, 0.914146, 0.943061, 0.965525, 0.986655, 0.985988, 0.989769, 0.990881],
    train_precision_list=None,
    train_f1_score_list=None,
    validation_acc_list=[
        0.7875, 0.842708, 0.85, 0.864583, 0.868958, 0.877917, 0.872708, 0.881458, 0.869375, 0.839583],
    validation_loss_list=[0.4664, 0.3931, 0.3759, 0.3742, 0.3529, 0.3524, 0.3463, 0.3439, 0.3813, 0.4025],
    validation_recall_list=[
        0.528125, 0.625, 0.854167, 0.842708, 0.934375, 0.913542, 0.948958, 0.920833, 0.940625, 0.966667],
    validation_precision_list=None,
    validation_f1_score_list=None
)
plot_result(eval_result=eval_res4, res_name="result4-bert-uncased-2e-5-70-15-15")
