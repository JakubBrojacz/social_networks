import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sklearn.linear_model

NUM_BINS = 30

def power_dist_analysis(x):

    plt.subplot(411)
    hist, bins, _ = plt.hist(x, bins=NUM_BINS)

    hist_x = np.asarray([(bins[i+1] + bins[i])/2 for i in range(NUM_BINS)])
    hist_x = hist_x.reshape(-1, 1)
    hist_y = hist
    reg = sklearn.linear_model.LinearRegression().fit(hist_x, hist_y)
    plt.plot([0, bins[-1]], [reg.predict([[0]]), reg.predict([[bins[-1]]])])
    plt.text(0.1*bins[-1], 0.8*max(hist), f'{reg.coef_[0]:.4f}x + {reg.intercept_:.4f}')

    plt.subplot(412)
    hist, bins, _ = plt.hist(x, bins=NUM_BINS)

    regression_x = []
    regression_y = []
    hist_x = np.asarray([(bins[i+1] + bins[i])/2 for i in range(NUM_BINS)])
    for i in range(NUM_BINS):
        if hist[i] > 0:
            regression_x.append([np.log10(hist_x[i])])
            regression_y.append(np.log10(hist[i]))
    reg = sklearn.linear_model.LinearRegression().fit(regression_x, regression_y)
    plt.plot([1, bins[-1]],
        [np.float_power(10, reg.predict([[0]])), np.float_power(10, reg.predict([[np.log10(bins[-1])]]))])
    plt.text(0.1*bins[-1], 0.5*max(hist), f'{reg.coef_[0]:.4f}x + {reg.intercept_:.4f}')

    plt.xscale('log')
    plt.yscale('log')

    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.subplot(413)
    hist, bins, _ = plt.hist(x, bins=logbins)

    regression_x = []
    regression_y = []
    hist_x = np.asarray([(bins[i+1] + bins[i])/2 for i in range(NUM_BINS)])
    for i in range(NUM_BINS):
        if hist[i] > 0:
            regression_x.append([np.log10(hist_x[i])])
            regression_y.append(np.log10(hist[i]))
    reg = sklearn.linear_model.LinearRegression().fit(regression_x, regression_y)
    plt.plot([1, bins[-1]],
        [np.float_power(10, reg.predict([[0]])), np.float_power(10, reg.predict([[np.log10(bins[-1])]]))])
    plt.text(0.1*bins[-1], 0.5*max(hist), f'{reg.coef_[0]:.4f}x + {reg.intercept_:.4f}')

    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(414)
    plt.xscale('log')
    plt.yscale('log')
    unique_points, counts = np.unique(x.astype(np.int32), return_counts=True)
    probs = counts / np.sum(counts)
    rv = scipy.stats.rv_discrete(values=(unique_points, probs))
    plt.plot(unique_points, rv.sf(unique_points))

    regression_x = np.log10(unique_points)[:-1].reshape(-1, 1)
    regression_y = np.log10(rv.sf(unique_points)[:-1])
    reg = sklearn.linear_model.LinearRegression().fit(regression_x, regression_y)
    plt.plot([1, unique_points[-2]],
        [np.float_power(10, reg.predict([[0]])), np.float_power(10, reg.predict([[np.log10(unique_points[-2])]]))])
    plt.text(0.1*unique_points[-2], 0.5, f'{reg.coef_[0]:.4f}x + {reg.intercept_:.4f}')

    plt.show()

    X_MIN = 1

    mle_estimator = x.shape[0] / (np.sum(np.log(x)) - np.log(X_MIN)) + 1
    print(f"mle estimator alpha: {mle_estimator}")


if __name__ == '__main__':

    n = 10000
    x = np.random.zipf(2., n)
    power_dist_analysis(x)
