import numpy as np
import pandas as pd


def norm_to_1(weight):
    _sum_ = sum(weight)
    norm_weight = np.array([arr/_sum_*1 for arr in weight])
    return norm_weight

def norm_to_0and1(array):
    _max_ = max(array)
    norm_array = np.array([arr/_max_ for arr in array])
    return norm_array

def normalize(array):
    _ave_ = np.average(array)
    _var_ = np.var(array)
    norm_array = np.array([(arr - _ave_) / _var_ for arr in array])
    return norm_array


def getsum_score(score, weight):
    all_score = np.array(score[0]) * weight[0]
    for i in range(1, len(score)):
        all_score = all_score + np.array(score[i]) * weight[i]
    return all_score

def score2label_threshold(score, percentage=0.95):
    import math
    score = np.array(score)
    temp_array = score.copy()
    temp_array.sort()
    if percentage == 0:
        threshold = temp_array[0]
    else:
        threshold = temp_array[math.floor(len(temp_array) * percentage)-1]

    label = np.zeros(len(score), dtype=int)
    for ii, s in enumerate(score):
        if s > threshold:
            label[ii] = 1
    return label

def read_data(path):
    csvdata = pd.read_csv(path, engine='python', usecols=['timestamp', 'value', 'label'])
    # csvdata = csvdata[:50000]
    timestamp = csvdata['timestamp'].values
    array_data = csvdata['value'].values
    real_result = csvdata['label'].values
    return array_data, real_result, timestamp

def write(path, text, function='a'):
    f1 = open(path, function)
    f1.write(text)
    f1.close()

if __name__ == '__main__':
    path = "D:/0学习/0毕设/数据/kpi/kpi_0.csv"
    print(read_data(path))