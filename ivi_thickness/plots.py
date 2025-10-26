import matplotlib.pyplot as plt
import numpy as np

def scatter_with_fit(x, y, yhat, title, xlabel, ylabel):
    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.7)
    plt.plot(np.sort(x), yhat[np.argsort(x)], lw=2)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()

def residual_hist(resid, title):
    plt.figure()
    plt.hist(resid, bins=20, alpha=0.8)
    plt.title(title); plt.xlabel("residual"); plt.ylabel("count")
    plt.tight_layout()
