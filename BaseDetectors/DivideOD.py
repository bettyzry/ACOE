import numpy as np
import util
import scipy.stats as stats
import statsmodels.api as sm
import sesd
from pyculiarity import detect_ts


class DivideOD:
    def __init__(self, method_score2label="three_sigma",
                 threshold_percentage=0.95, max_feed_len=10000, max_store_len=1000000):
        self.threshold_percentage = threshold_percentage
        self.max_feed_len = max_feed_len
        self.max_store_len = max_store_len
        self.history = np.empty((0, ))
        self.past_data = np.empty([0, ])
        self.past_len = 0
        self.feed_data = np.empty([0, ])
        self.method_score2label = method_score2label
        return


    def predict(self, data, seasonal=None):
        all_score = np.zeros(len(data))
        result = np.zeros(len(data), dtype=int)
        seasonal2 = 0
        return all_score, result, seasonal2

    def fit(self, input_data, seasonal=None):
        # prepare data for model feeding

        temp_data = np.concatenate((self.past_data, input_data), axis=0)
        temp_len = len(temp_data)

        if len(input_data) > self.max_feed_len:
            self.feed_data = input_data
        elif temp_len < self.max_feed_len:
            self.feed_data = temp_data
        else:
            self.feed_data = temp_data[-self.max_feed_len:]
        # change the length of a data
        all_score, result, seasonal2 = self.predict(data=self.feed_data, seasonal=seasonal)
        anomaly_indices = np.where(result == 1)[0]
        anomaly_score = all_score[anomaly_indices]

        # store
        if len(temp_data) < self.max_store_len:
            self.past_data = temp_data
        else:
            self.past_data = input_data

        return all_score, anomaly_score, anomaly_indices, result, seasonal2


class SESD(DivideOD):
    def __init__(self, method_score2label="three_sigma",
                 threshold_percentage=0.95, max_feed_len=10000, max_store_len=1000000):
        DivideOD.__init__(self, method_score2label, threshold_percentage, max_feed_len, max_store_len)
        return

    def predict(self, data, seasonal=None):
        diff_value, seasonal2 = self.seasonal_esd(data, seasonality=seasonal, max_anomalies=self.max_feed_len)
        # diff_value, result, seasonal2 = self.seasonal_esd(data, max_anomalies=self.max_feed_len, alpha=self.threshold_percentage)
        diff_value_normalize = util.normalize(diff_value)

        label = util.score2label_threshold(diff_value_normalize, percentage=self.threshold_percentage)
        # sesd.seasonal_esd(data, hybrid=False, max_anomalies=10, alpha=0.05)
        return abs(diff_value_normalize), label, seasonal2


    def seasonal_esd(self, data, seasonality=None, hybrid=False, max_anomalies=10, alpha=0.05):
        ts = np.array(data)
        seasonal = seasonality or int(0.2 * len(ts))  # Seasonality is 20% of the ts if not given.

        # print(seasonal, int(0.2 * len(ts)))

        decomp = sm.tsa.seasonal_decompose(ts, model="additive", freq=seasonal)
        residual = ts - decomp.seasonal - np.median(ts)
        # print(np.median(ts))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(411)
        # plt.ylabel('ts2')
        # plt.plot(ts)
        # plt.subplot(412)
        # plt.ylabel('trend')
        # plt.plot(decomp.trend)
        # plt.subplot(413)
        # plt.ylabel('seasonal')
        # plt.plot(decomp.seasonal)
        # plt.subplot(414)
        # plt.ylabel('resid')
        # # plt.ylim(-1, 1)
        # plt.plot(residual)
        # plt.show()
        # outliers, _ = self.esd(residual, max_anomalies=max_anomalies, alpha=alpha, hybrid=hybrid)
        # zscores = abs(calculate_zscore(residual, hybrid=hybrid))
        # outliers = Utils.score2label_threshold(zscores, self.threshold_percentage)
        return residual, decomp.seasonal

    def esd(self, timeseries, max_anomalies, alpha=0.05, hybrid=False):
        ts = np.copy(np.array(timeseries))
        test_statistics = []
        test_values = []
        total_anomalies = -1
        max_anomalies = int((1-self.threshold_percentage)*len(timeseries))
        # print(max_anomalies)
        # max_anomalies = len(timeseries)
        # print(max_anomalies, self.threshold_percentage)
        for curr in range(max_anomalies):
            test_idx, test_val = sesd.calculate_test_statistic(ts, test_statistics, hybrid=hybrid)
            critical_value = sesd.calculate_critical_value(len(ts) - len(test_statistics), alpha)
            if test_val > critical_value:
                total_anomalies = curr
            test_statistics.append(test_idx)
            test_values.append(test_val)

        anomalous_indices = test_statistics[:total_anomalies + 1]
        result = [1 if i in anomalous_indices else 0 for i in range(0,len(timeseries))]
        result = np.array(result)
        # return result       # anomalous_indices
        return result, test_values


def calculate_zscore(ts, hybrid=False):
    if hybrid:
        median = np.median(ts)
        mad = np.median(np.abs(ts - median))
        return (ts - median) / mad
    else:
        return stats.zscore(ts, ddof=1)

def scan_anomaly_sesd(ts_data, threshold_percentage=0.95, seasonal=1500):#1500
    sesd = SESD(threshold_percentage=threshold_percentage, max_feed_len=1000, max_store_len=1000000)
    all_score, anomaly_score, anomaly_indices, result, seasonal2 = sesd.fit(ts_data, seasonal=seasonal)
    return all_score