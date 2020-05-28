import util
import evaluate

class Zscore:
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        return

    def predict(self, data):
        diff_value_normalize = abs(util.normalize(data))    # 标准化
        result = util.score2label_threshold(score=diff_value_normalize, percentage=self.threshold)
        return diff_value_normalize, result

def scan_anomaly_zscore(ts_data, threshold):
    zscore = Zscore(threshold)
    score, y_pre = zscore.predict(ts_data)
    return score

if __name__ == '__main__':
    kpi = 0
    path = '../data/kpi/kpi_' + str(kpi) + '.csv'
    data, y_true, timestamp = util.read_data(path)
    threshold = 1 - sum(y_true)/len(y_true)
    score = scan_anomaly_zscore(data, threshold)
    precision, recall, f1, roc, pr, precision_eval, recall_eval, f1_eval, newresult, text = evaluate.evaluate('name', True, score, y_true, threshold)
    print(text)