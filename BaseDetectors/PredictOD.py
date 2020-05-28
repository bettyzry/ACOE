# @Time : 2019/7/29 18:25
# @Author : Hongzuo Xu
# @Description ：
# Implementations of some prediction-based outlier detection methods for time-series data.

from __future__ import division
import numpy as np
import util
import evaluate


class PredictOD:
    def __init__(self, win_size=10, n=10, threshold=0.99):
        self.threshold_percentage = threshold
        self.win_size = win_size
        self.n = n
        self.history = np.empty((0, ))
        return

    def predict(self, data):
        all_score = np.zeros(len(data))
        result = np.zeros(len(data), dtype=int)
        return all_score, result



class HoltWinters(PredictOD):
    def __init__(self, alpha=0.5, beta=0.5, types='linear', gama=0.5, m=1,
                 win_size=10, threshold=0.99):
        '''
        :param alpha, beta: for three types
        :param types: linear 、additive 、multiplicative
        :param gama: for additive and multiplicative
        :param m: for additive and multiplicative
        '''
        PredictOD.__init__(self, win_size, m,  threshold)
        self.alpha = alpha
        self.beta = beta
        self.types = types
        self.gama = gama
        self.m = m
        self.history = np.empty((0, ))

    def predict(self, data):
        Y = data
        types = self.types

        if types == 'linear':
            alpha, beta = self.alpha, self.beta
            a = [Y[0]]
            b = [Y[1] - Y[0]]
            y = [a[0] + b[0]]
            for i in range(len(Y)):
                a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                y.append(a[i + 1] + b[i + 1])
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gama
            m = self.m
            a = [sum(Y[0:m]) / float(m)]
            b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]

            if types == 'additive':
                s = [Y[i] - a[0] for i in range(m)]
                # print(a,b,s)
                y = [a[0] + b[0] + s[0]]
                for i in range(len(Y)):
                    a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
                    b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                    s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
                    y.append(a[i + 1] + b[i + 1] + s[i + 1])

            elif types == 'multiplicative':
                s = [Y[i] / a[0] for i in range(m)]
                y = [(a[0] + b[0]) * s[0]]
                for i in range(len(Y)):
                    a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
                    b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                    s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
                    y.append((a[i + 1] + b[i + 1]) * s[i + 1])
            else:
                raise ValueError("ERROR: unsupported type, expect linear, additive or multiplicative.")
        y.pop()

        diff_value = abs(data - y)
        diff_value_normalize = util.normalize(diff_value)
        result = util.score2label_threshold(diff_value_normalize, percentage=self.threshold_percentage)
        return diff_value_normalize, result

class MA(PredictOD):
    def __init__(self, win_size=10, n=10, threshold=0.95):
        PredictOD.__init__(self, win_size, n, threshold)
        return

    def predict(self, data):
        win_size = self.win_size
        res = np.array([np.average(data[i: i + win_size]) for i in range(len(data) - win_size)])
        predict_value = np.concatenate((data[:win_size], res))

        diff_value = abs(data - predict_value)
        diff_value_normalize = util.normalize(diff_value)
        result = util.score2label_threshold(diff_value_normalize, percentage=self.threshold_percentage)
        return diff_value_normalize, result


class EWMA(PredictOD):
    def __init__(self, beta=0.9, win_size=10, n=10, threshold=0.95):
        '''
        :param beta: EWMA parameter
        '''
        PredictOD.__init__(self, win_size, n, threshold)
        self.beta = beta
        return

    def predict(self, data):
        beta = self.beta
        predict_value = np.zeros(len(data))
        predict_value[0] = data[0]
        for i in range(1, len(data)):
            predict_value[i] = beta * predict_value[i - 1] + (1 - beta) * data[i]

        diff_value = abs(data - predict_value)
        diff_value_normalize = util.normalize(diff_value)
        result = util.score2label_threshold(diff_value_normalize, percentage=self.threshold_percentage)

        return diff_value_normalize, result

def scan_anomaly_ma(ts_data, threshold):
    ma = MA(threshold=threshold)
    score, y_pre = ma.predict(ts_data)
    return score

def scan_anomaly_ewma(ts_data, threshold):
    ma = EWMA(threshold=threshold)
    score, y_pre = ma.predict(ts_data)
    return score

def scan_anomaly_holtwinter(ts_data, threshold):
    ma = HoltWinters(threshold=threshold)
    score, y_pre = ma.predict(ts_data)
    return score

if __name__ == '__main__':
    kpi = 0
    path = '../data/kpi/kpi_' + str(kpi) + '.csv'
    data, y_true, timestamp = util.read_data(path)
    threshold = 1 - sum(y_true)/len(y_true)
    score = scan_anomaly_holtwinter(data, threshold)
    precision, recall, f1, roc, pr, precision_eval, recall_eval, f1_eval, newresult, text = evaluate.evaluate('name', True, score, y_true, threshold)
    print(text)