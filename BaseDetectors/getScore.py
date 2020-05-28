from BaseDetectors.DivideOD import scan_anomaly_sesd
from BaseDetectors.NormalOD import scan_anomaly_zscore
from BaseDetectors.PredictOD import scan_anomaly_ma, scan_anomaly_ewma, scan_anomaly_holtwinter
import numpy as np

def GetScore(ts_data, threshold, seasonal):
    score = []
    score.append(scan_anomaly_ma(ts_data, threshold))
    score.append(scan_anomaly_ewma(ts_data, threshold))
    score.append(scan_anomaly_holtwinter(ts_data, threshold))
    score.append(scan_anomaly_zscore(ts_data, threshold))
    score.append(scan_anomaly_sesd(ts_data, threshold, seasonal))
    score = np.array(score)
    return score