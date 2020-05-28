import numpy as np
import util
import evaluate
import statsmodels.api as sm


class SESD:
    def __init__(self, threhold=0.99, seasonal=None):
        self.threshold = threhold
        self.seasonal = seasonal
        return

    def predict(self, data,):
        diff_value = self.seasonal_esd(data, seasonality=self.seasonal)
        diff_value_normalize = util.normalize(abs(diff_value))
        label = evaluate.score2label_threshold(diff_value_normalize, percentage=self.threshold)
        return abs(diff_value_normalize), label

    def seasonal_esd(self, data, seasonality=None):
        ts = np.array(data)
        seasonal = seasonality or int(0.2 * len(ts))  # Seasonality is 20% of the ts if not given.
        decomp = sm.tsa.seasonal_decompose(ts, model="additive", freq=seasonal)
        residual = ts - decomp.seasonal - np.median(ts)
        return residual

def scan_anomaly_sesd(ts_data, threshold, seasonal=None):
    sesd = SESD(threhold=threshold, seasonal=seasonal)
    score, y_pre = sesd.predict(ts_data)
    return score

if __name__ == '__main__':
    kpi = 0
    path = '../data/kpi/kpi_' + str(kpi) + '.csv'
    data, y_true, timestamp = util.read_data(path)
    threshold = 1 - sum(y_true)/len(y_true)
    score = scan_anomaly_sesd(data, threshold)
    precision, recall, f1, roc, pr, precision_eval, recall_eval, f1_eval, newresult, text = evaluate.evaluate('name', True, score, y_true, threshold)
    print(text)