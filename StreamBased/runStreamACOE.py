from PoolBased.runPoolACOE import ColdStart
from StreamBased.StreamACOE import StreamACOE
import util
import evaluate
import pandas as pd
import numpy as np

def countfeedback(feedback):
    arr_feedback = np.array(feedback)
    a = arr_feedback[arr_feedback != -1]
    size = len(a)
    return size

def scan_anomaly_streamacoe(ts_data, y_true, threshold, coldstart=10000, win=100000, batch=64, query_rate=0.02, seasonal=1440, dataset='dontkonw'):
    pathout = str(dataset) + '_result' + str(win) +'.csv'
    util.write(pathout, ',precision, recall, f1, roc, pr, precision_eval, recall_eval, f1_eval, samplenum\n', 'w')
    num_feedback = 0
    weight, feedback, y_pre = ColdStart(ts_data[:coldstart], y_true[:coldstart], threshold, seasonal)
    num_feedback += countfeedback(feedback)
    acoe = StreamACOE(weight=weight, winsize=win, threshold=threshold, seasonal=seasonal, history=ts_data[:coldstart],
                      feedback=feedback, query_rate=query_rate, batch=batch)

    feedback = np.array([-1] * batch)
    for i in range(coldstart, len(ts_data), batch):
        if i+batch > len(ts_data):
            break
        data = ts_data[i: i+batch]
        label = y_true[i: i + batch]
        is_anomaly, need_labeled, is_stresstest = acoe.fit(data, feedback)
        y_pre = np.concatenate([y_pre, is_anomaly])
        feedback = [label[i] if need_labeled[i] == 1 else -1 for i in range(batch)]
        num_feedback += countfeedback(feedback)
        acoe.update_feedback(feedback)
        acoe.update_weight()
        print(acoe.weight)
        temp_y_true = y_true[:len(y_pre)]
        precision, recall, f1, roc, pr, precision_eval, recall_eval, f1_eval, newresult, text = evaluate.evaluate(i,
                                                                                                                  False,
                                                                                                                  y_pre,
                                                                                                                  temp_y_true,
                                                                                                                  threshold,
                                                                                                                  str(num_feedback))
        util.write(pathout, text)
        print(text)
    return y_pre


def scan_anomaly_stream(ts_data, y_true, threshold, coldstart=10000, win=100000, batch=64, query_rate=0.0005, seasonal=1440, dataset='dontkonw'):
    def predict(data, threshold):
        weight = [0.2,0.2,0.2,0.2,0.2]
        from BaseDetectors.getScore import GetScore
        score = GetScore(data, threshold, seasonal)
        all_score = np.dot(weight, score)
        y_pre = util.score2label_threshold(all_score, threshold)
        return y_pre

    pathout = str(dataset) + '_result_noacoe' + str(win) +'.csv'
    util.write(pathout, ',precision, recall, f1, roc, pr, precision_eval, recall_eval, f1_eval\n', 'w')

    y_pre = predict(ts_data[:coldstart], threshold)
    for i in range(coldstart, len(ts_data), batch):
        if i + batch > len(ts_data):
            break
        data = ts_data[:i + batch]
        is_anomaly = predict(data, threshold)
        y_pre = np.concatenate([y_pre, is_anomaly[-batch:]])
        temp_y_true = y_true[:len(y_pre)]
        precision, recall, f1, roc, pr, precision_eval, recall_eval, f1_eval, newresult, text = evaluate.evaluate(i,
                                                                                                                  False,
                                                                                                                  y_pre,
                                                                                                                  temp_y_true,
                                                                                                                  threshold)
        util.write(pathout, text)
        print(text)
    return y_pre


if __name__ == '__main__':
    kpilist = [0,4,5,8,11,15,16,18,20,23,24,25]
    batchlist = [32,64,128,256,512,1024]
    winlist = [20000, 40000, 80000, 160000]
    kpi = 4
    coldstart = 10000
    win = 100000
    batch = 64
    query_rate = 0.02
    seasonal = 1440
    for win in winlist:
        print('################ kpi_' + str(kpi) +' ###################')
        path = '../data/kpi/kpi_' + str(kpi) + '.csv'
        df = pd.read_csv(path)
        ts_data = df['value'].values
        y_true = df['label'].values
        threshold = 1 - sum(y_true) / len(y_true)/4



        y_pre = scan_anomaly_streamacoe(ts_data, y_true, threshold, coldstart, win, batch, query_rate, seasonal, str(kpi))
        # y_pre = scan_anomaly_stream(ts_data, y_true, threshold, coldstart, win, batch, query_rate, seasonal, str(kpi))
        y_true = y_true[:len(y_pre)]

        precision, recall, f1, roc, pr, precision_eval, recall_eval, f1_eval, newresult, text = evaluate.evaluate(kpi,
                                                                                                                  False,
                                                                                                                  y_pre,
                                                                                                                  y_true,
                                                                                                                  threshold)
        print(text)
