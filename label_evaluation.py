import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import copy as cp
import sys


def label_evaluation(y_true, y_pred, delay1=3, delay2=5):
    if len(y_pred) != len(y_true):
        print('\033[31mERROR: length is not equal for y_true and y_pred')
        sys.exit(1)

    splits = np.where(y_true[1:] != y_true[:-1])[0] + 1
    new_predict = cp.copy(y_pred)

    length = len(y_true)
    l_splits = len(splits)

    is_anomaly = y_true[0] == 1

    if is_anomaly:
        t2 = min(length, splits[0]+delay1+1)
        if 1 in y_pred[0: t2]:
            new_predict[0:splits[0]] = 1

    is_anomaly = not is_anomaly

    for i in range(l_splits):
        if is_anomaly:
            start = splits[i]
            t1 = max(0, start - delay2)

            if i == l_splits - 1:
                t2 = length
                end = length
            else:
                t2 = min(length, splits[i + 1] + delay1 + 1)
                end = splits[i + 1]

            if 1 in y_pred[t1:t2]:
                new_predict[start:end] = 1

        is_anomaly = not is_anomaly

    fscore = f1_score(y_true, new_predict)
    p = precision_score(y_true, new_predict)
    r = recall_score(y_true, new_predict)
    # print('{:.4},{:.4},{:.4}'.format(p, r, fscore))
    return new_predict



if __name__ == '__main__':
    # y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # y_true = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1])
    # y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    y_true = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    new_pred = label_evaluation(y_true=y_true, y_pred=y_pred)
    print('true:  ', y_true)
    print('pred:  ', y_pred)
    print('new_p: ', new_pred)
    # fscore = f1_score(y_true, new_pred)
    # p = precision_score(y_true, new_pred)
    # r = recall_score(y_true, new_pred)
    # print(p, r, fscore)