import math


def get_values(cm):
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    recall = round(TP / (TP + FN), 2)
    specificity = round(TN / (TN + FP), 2)
    precision = round(TP / (TP + FP), 2)
    npv = round(TN / (TN + FN), 2)
    f1 = 2 * (precision * recall) / (precision + recall)
    mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
    return precision, recall, f1
