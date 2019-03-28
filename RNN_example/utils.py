import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def get_time_series(data, num_periods=7, f_horizon=1, scale=False):    
    n, m = data.shape

    x_data = data[:(len(data) - (len(data) % num_periods))]
    y_data = data[1:(len(data)-(len(data) % num_periods)) + f_horizon]
    
    if scale:
        std_scale = preprocessing.StandardScaler().fit(x_data, y_data)
        x_data = std_scale.transform(x_data)
        y_data = std_scale.transform(y_data)
    
    
    
    x_batches = x_data.reshape(-1, num_periods, m)
    y_batches = y_data.reshape(-1, num_periods, m)
    
    if scale:
        return x_batches, y_batches, std_scale
    else:
        return x_batches, y_batches


def plot(y_samples, y_test,title_name="No title",file=None):
    samples = y_samples.shape[0]
    if True:
        plt.figure(figsize=(8,6))
    for i in range(samples):
        plt.plot(pd.Series(np.ravel(y_samples[i])), "b-", alpha=1. / 200)
    plt.plot(pd.Series(np.ravel(y_test)), "ro", markersize=10, label="Actual")
    plt.title(title_name)
    plt.xlabel("Months")
    plt.ylabel("Resistance")
    if file:
        plt.savefig("./temp/fig/"+str(file))
