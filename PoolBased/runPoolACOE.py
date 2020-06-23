from PoolBased.PoolACOE import PoolACOE
from BaseDetectors.DivideOD import scan_anomaly_sesd
from BaseDetectors.NormalOD import scan_anomaly_zscore
from BaseDetectors.PredictOD import scan_anomaly_ma, scan_anomaly_ewma, scan_anomaly_holtwinter
import util
import evaluate
import numpy as np
from plot import plot


def scan_anomaly_poolacoe(ts_data, label, threshold, seasonal=None, dropstress=False):
    score = []
    score.append(scan_anomaly_ewma(ts_data, threshold))
    score.append(scan_anomaly_ma(ts_data, threshold))
    score.append(scan_anomaly_holtwinter(ts_data, threshold))
    score.append(scan_anomaly_zscore(ts_data, threshold))
    score.append(scan_anomaly_sesd(ts_data, threshold, seasonal))
    acoe = PoolACOE(threshold=threshold)
    y_pre, y_stress, score, score_nostress = acoe.predict(score, label, dropstress)
    return y_pre, y_stress, score, score_nostress


def ColdStart(ts_data, label, threshold, seasonal=None):
    score = []
    score.append(scan_anomaly_ewma(ts_data, threshold))
    score.append(scan_anomaly_ma(ts_data, threshold))
    score.append(scan_anomaly_holtwinter(ts_data, threshold))
    score.append(scan_anomaly_zscore(ts_data, threshold))
    score.append(scan_anomaly_sesd(ts_data, threshold, seasonal))
    acoe = PoolACOE(threshold=threshold)
    y_pre, y_stress, diff_value_normalize, nostress_score = acoe.predict(score, label)
    queriedindex = np.array(acoe.queried).T[0]  # [[index,label],[index, label]]
    feedback = [label[i] if i in queriedindex else -1 for i in range(len(ts_data))]
    return acoe.weight, feedback, y_pre

def show_stressaeod():
    kpi = 16
    path = '../data/presure/kpi_' + str(kpi) + '.csv'
    data, y_complex, timestamp = util.read_data(path)
    index_true = np.where(y_complex == 1)[0]
    index_presure = np.where(y_complex == 2)[0]
    y_true = np.zeros(len(y_complex))
    y_true[index_true] = 1
    y_presure = np.zeros(len(y_complex))
    y_presure[index_presure] = 1

    threshold = 1 - (sum(y_true) + sum(y_presure)) / len(y_true)
    seasonal = 1440
    y_pre, y_presure_pre, score, score_nostress = scan_anomaly_poolacoe(data, y_true, threshold, seasonal, True)
    threshold = 1 - sum(y_true) / len(y_true)
    _, _, _, _, _, _, _, _, _, text = evaluate.evaluate('name', True, score, y_true, threshold)
    print('without drop stress', text)
    _, _, _, _, _, _, _, _, y_pre_eval, text = evaluate.evaluate('name', False, y_pre, y_true, threshold)
    print('drop stress', text)
    _, _, _, _, _, _, _, _, y_presure_pre_eval, text = evaluate.evaluate('name', False, y_presure_pre, y_presure,
                                                                         threshold)
    print('stress', text)

    labels = [y_true, y_pre_eval, y_presure, y_presure_pre_eval]
    title = ['outlier', 'predicted outlier', 'presure test', 'predicted presure test']
    color = ['ro', 'go', 'ro', 'go']
    plot(data, timestamp, labels, title, color)


def show_aeod():
    kpi = 0
    for kpi in range(11):
        path = '../data/kpi/d/kpi_' + str(kpi) + '.csv'
        data, y_true, timestamp = util.read_data(path)
        threshold = 1 - (sum(y_true)) / len(y_true)
        seasonal = 1440
        y_pre, y_presure_pre, score, score_nostress = scan_anomaly_poolacoe(data, y_true, threshold, seasonal, False)
        _, _, _, _, _, _, _, _, _, text = evaluate.evaluate('kpi'+str(kpi), True, score, y_true, threshold)
        print(text)

if __name__ == '__main__':
    # show_stressaeod()
    show_aeod()

