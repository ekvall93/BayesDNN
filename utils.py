import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
def plot(Y_sample, X_pred, X, y, samples, color_line="b-",path=None):
    if True:
    #    plt.figure(figsize=(8,6))
        for i in range(samples):
            plt.plot(X_pred[:, 0], Y_sample[i], color_line, alpha=1. / 200)
        plt.plot(X[:, 0], y, "r.")
        if path:
            plt.savefig(path + ".jpg")
        
