import numpy as np
import util


def scan_anomaly_zscore(ts_data, threshold_percentage=0.99, quired=None, kpi=1):
    zscore = ZScore(f_win_size=1000, method_score2label="threshold",   #threshold, three_sigma
                        threshold_percentage=threshold_percentage, max_feed_len=10000, max_store_len=1000000, quired=quired)

    all_score, anomaly_score, anomaly_indices, y_zscore = zscore.fit(ts_data)
    return all_score

class NormalOD:
    def __init__(self, f_win_size=10, method_score2label="three_sigma",
                 threshold_percentage=0.95, max_feed_len=10000, max_store_len=1000000, quired=None):
        self.method_score2label = method_score2label
        self.threshold_percentage = threshold_percentage
        self.max_feed_len = max_feed_len
        self.max_store_len = max_store_len
        self.history = np.empty((0, ))
        self.f_win_size = f_win_size

        self.quired = quired
        self.past_data = np.empty([0, ])
        self.past_data_m = np.empty([0, self.f_win_size])
        self.past_len = 0

        self.feed_data = np.empty([0, ])
        self.feed_data_m = np.empty([0, ])

        return

    def feature_extraction(self, org_data):
        '''
        using past #window_size data to represent current value
        :param org_data:
        :return: data matrix with shape (len(org_data) - window_size + 1, feature_num)
        '''
        n_f_vector = len(org_data)-self.f_win_size+1
        data_matrix = np.zeros([n_f_vector, self.f_win_size])
        for i in range(n_f_vector):
            data_matrix[i] = org_data[i:i+data_matrix.shape[1]]
        return data_matrix

    def score2label(self, y_scores):
        label = util.score2label_threshold(y_scores, percentage=self.threshold_percentage)
        return label

    def predict(self, data):
        all_score = np.zeros(len(data))
        result = np.zeros(len(data), dtype=int)
        return all_score, result

    def fit(self, input_data):
        feed_data = np.copy(input_data)
        all_score, result = self.predict(feed_data)

        anomaly_indices = np.where(result == 1)[0]
        anomaly_score = all_score[anomaly_indices]

        return all_score, anomaly_score, anomaly_indices, result



class ZScore(NormalOD):
    def __init__(self, f_win_size=10, method_score2label="three_sigma",
                 threshold_percentage=0.95, max_feed_len=10000, max_store_len=1000000, quired=None):
        NormalOD.__init__(self, f_win_size, method_score2label, threshold_percentage, max_feed_len, max_store_len)
        self.ave = 0
        self.tempsum = 0
        self.quired = quired
        return

    def train(self, data):
        true_data = np.copy(data)
        # if self.quired is not None:
        #     for i in self.quired:
        #         if i[1] == 1:
        #             true_data = np.delete(true_data, i[0], axis=0)
        lenth = len(true_data)
        total = sum(true_data)
        self.ave = float(total)/lenth
        self.tempsum = sum([pow(data[i] - self.ave,2) for i in range(lenth)])
        self.tempsum = pow(float(self.tempsum)/lenth,0.5)

    def predict(self, data):
        self.train(data)
        diff_value = self.Z_Score(data)
        diff_value_normalize = util.normalize(abs(diff_value))
        result = util.score2label_threshold(score=diff_value_normalize, percentage=self.threshold_percentage)
        return diff_value_normalize, result

    def Z_Score(self, data):
        value = np.zeros(len(data))
        # print(self.ave, self.tempsum)
        for i in range(len(data)):
            value[i] = (data[i] - self.ave) / self.tempsum
        return value