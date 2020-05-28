import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(data, timestamp, labels, title=None, color=None):
    time = pd.to_datetime(timestamp)
    plt.title('kpi_16')
    for i in range(len(labels)):
        index = np.where(labels[i] == 1)[0]
        time_point = time[index]
        data_point = data[index]
        plt.subplot(len(labels), 1, i+1)
        plt.ylabel(title[i])
        plt.plot(time, data, '-', color='#0066CC')
        plt.plot(time_point, data_point, color[i])
    plt.grid(True)
    plt.show()
