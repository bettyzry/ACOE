# @Time : 2019/7/29 18:25
# @Author : Hongzuo Xu
# @Description ：
# Implementations of some prediction-based outlier detection methods for time-series data.

from __future__ import division
import numpy as np
import util


class PredictOD:
    def __init__(self, win_size=10, n=10,
                 method_score2label="three_sigma", threshold_percentage=0.95, quired=None):
        self.method_score2label = method_score2label
        self.threshold_percentage = threshold_percentage
        self.win_size = win_size
        self.n = n
        self.history = np.empty((0, ))
        self.quired = quired

        return

    def predict(self, data):
        all_score = np.zeros(len(data))
        result = np.zeros(len(data), dtype=int)
        return all_score, result

    def fit(self, input_data):
        temp = np.concatenate((self.history, np.array(input_data)))
        new_length = len(input_data)

        if 0 <= len(self.history) <= self.n * self.win_size:
            # history_data + new input < win_size, then report all of fitted data as normal data
            if (new_length + len(self.history)) <= self.win_size:
                result = np.zeros(new_length, dtype=int)
                all_score = np.zeros(new_length)
            # history_data + new input >= win_size, then use historty+new_data to fit model
            else:
                all_score, result = self.predict(temp)
        # history data are too large, only using n*win_size history data + new_data to fit model
        else:
            all_score, result = self.predict(temp[-(self.n * self.win_size + new_length):])

        result = result[-new_length:]
        all_score = all_score[-new_length:]

        if len(temp) < 500000:
            self.history = temp
        else:
            self.history = input_data

        anomaly_indices = np.where(result == 1)[0]
        anomaly_score = all_score[anomaly_indices]

        return all_score, anomaly_score, anomaly_indices, result


class HoltWinters(PredictOD):
    def __init__(self, alpha=0.5, beta=0.5, types='linear', gama=0.5, m=1,
                 win_size=10, n=10,
                 method_score2label="three_sigma", threshold_percentage=0.95):
        '''
        :param alpha, beta: for three types
        :param types: linear 、additive 、multiplicative
        :param gama: for additive and multiplicative
        :param m: for additive and multiplicative
        '''
        PredictOD.__init__(self, win_size, m, method_score2label, threshold_percentage)
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
        diff_value_normalize = util.normalize(abs(diff_value))
        result = util.score2label_threshold(score=diff_value_normalize, percentage=self.threshold_percentage)

        return diff_value_normalize, result

class MA(PredictOD):
    def __init__(self, win_size=10, n=10, method_score2label="three_sigma", threshold_percentage=0.95, quired=None):
        PredictOD.__init__(self, win_size, n, method_score2label, threshold_percentage)
        self.quired = quired
        return

    def predict(self, data):
        win_size = self.win_size
        evaldata = np.copy(data)
        if self.quired == None:
            res = np.array([np.average(data[i: i + win_size]) for i in range(len(data) - win_size)])
            predict_value = np.concatenate((data[:win_size], res))
        else:
            for item in self.quired:
                if item[1] == 1 and item[0] != 0: #anomaly
                    start = int(max(item[0] - win_size, 0))
                    evaldata[item[0]] = np.average(evaldata[start: item[0]])
            res = np.array([np.average(evaldata[i: i + win_size]) for i in range(len(data) - win_size)])
            predict_value = np.concatenate((evaldata[:win_size], res))
        diff_value = abs(data - predict_value)
        diff_value_normalize = util.normalize(abs(diff_value))
        result = util.score2label_threshold(score=diff_value_normalize, percentage=self.threshold_percentage)
        return diff_value_normalize, result


class EWMA(PredictOD):
    def __init__(self, beta=0.9, win_size=10, n=10, method_score2label="threshold", threshold_percentage=0.95):
        '''
        :param beta: EWMA parameter
        '''
        PredictOD.__init__(self, win_size, n, method_score2label, threshold_percentage)
        self.beta = beta
        return

    def predict(self, data):
        beta = self.beta
        predict_value = np.zeros(len(data))
        predict_value[0] = data[0]
        for i in range(1, len(data)):
            predict_value[i] = beta * predict_value[i - 1] + (1 - beta) * data[i]

        diff_value = abs(data - predict_value)
        diff_value_normalize = util.normalize(abs(diff_value))
        result = util.score2label_threshold(score=diff_value_normalize, percentage=self.threshold_percentage)

        return diff_value_normalize, result

def scan_anomaly_ma(ts_data, threshold_percentage=0.99, win_size=50, quired=None):
    ma = MA(win_size=win_size, n=10, method_score2label="threshold", threshold_percentage=threshold_percentage, quired=quired)
    all_score, anomaly_score, anomaly_indices, y_ma = ma.fit(ts_data)
    # ts_data = ts_data.values
    # ts_anomaly = ts_data[anomaly_indices]
    # ts_anomaly = []
    return all_score


def scan_anomaly_ewma(ts_data, threshold_percentage=0.99, win_size=100):
    ewma = EWMA(beta=0.9, win_size=win_size, n=10, method_score2label="threshold", threshold_percentage=threshold_percentage)
    all_score, anomaly_score, anomaly_indices, y_ewma = ewma.fit(ts_data)
    # ts_data = ts_data.values
    # ts_anomaly = ts_data[anomaly_indices]
    return all_score


def scan_anomaly_holtwinter(ts_data, threshold_percentage=0.99, types='linear', win_size=1, m=2):
    holt_winters = HoltWinters(alpha=0.5, beta=0.5, gama=0.5, types=types, win_size=win_size, n=10, m=m,
                               method_score2label="threshold", threshold_percentage=threshold_percentage)
    all_score, anomaly_score, anomaly_indices, y_holt = holt_winters.fit(ts_data)
    # ts_data = ts_data.values
    # ts_anomaly = ts_data[anomaly_indices]
    return all_score