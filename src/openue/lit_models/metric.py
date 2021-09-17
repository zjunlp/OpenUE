import numpy as np 
from openue.data import precision_score, recall_score, f1_score
from typing import Tuple, List, Dict


def compute_f1(logits, labels):
    n_gold = n_pred = n_correct = 0
    preds = np.argmax(logits, axis=-1)
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if pred != 0 and label != 0 and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        return {'precision': prec, 'recall': recall, 'f1': f1}


def acc(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return (preds == labels).mean()

# NER METRIC
def align_predictions(label_map_ner, predictions, label_ids) -> Tuple[List[int], List[int]]:

    predict = []
    label = []


    preds = np.argmax(predictions, axis=-1)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != -100:  # 特殊标签，还原句子的原有长度
                out_label_list[i].append(label_map_ner[label_ids[i][j]])
                preds_list[i].append(label_map_ner[preds[i][j]])

    predict.extend(preds_list)
    label.extend(out_label_list)

    return predict, label

def compute_metrics(predictions, label_ids, label_map_ner) -> Dict:
    preds_list, out_label_list = align_predictions(label_map_ner, predictions, label_ids)
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def seq_metric(preds, labels):
    def precision_score(y_true, y_pred):
        model_pred, labels = y_pred, y_true
        # 预测结果，大于这个阈值则视为预测正确
        accuracy_th = 0.5
        pred_result = model_pred > accuracy_th
        # pred_result = pred_result.float()
        pred_one_num = np.sum(pred_result)
        if pred_one_num == 0:
            return 0
        true_predict_num = np.sum(pred_result * labels)
        # 模型预测的结果中有多少个是正确的
        precision = true_predict_num / pred_one_num

        return precision.item()

    def recall_score(y_true, y_pred):
        model_pred, labels = y_pred, y_true
        # 预测结果，大于这个阈值则视为预测正确
        accuracy_th = 0.5
        pred_result = model_pred > accuracy_th
        # pred_result = pred_result.float()
        # numpy.ndarray
        pred_one_num = np.sum(pred_result)
        if pred_one_num == 0:
            return 0
        target_one_num = np.sum(labels)
        true_predict_num = np.sum(pred_result * labels)
        # 模型预测正确的结果中，占所有真实标签的数量
        recall = true_predict_num / target_one_num

        return recall.item()
    
    p = precision_score(labels, preds)
    r = recall_score(labels, preds)
    
    f1 = 2 * p * r / (p + r) if p and r else 0
    
    return dict(f1=f1, precision=p, recall=r)