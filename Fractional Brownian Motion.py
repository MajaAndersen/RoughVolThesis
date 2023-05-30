# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns

class fBm:
    def __init__(self, H, T, Tn):
        self.H = H
        self.T = T
        self.Tn = Tn
        self.dt = self.T / self.Tn
        self.times = [self.dt * n for n in range(1,self.Tn+1)]

    def simulate(self):
        cov = [[1 / 2 * (t ** (2 * self.H) + s ** (2 * self.H) - abs(t - s) ** (2 * self.H)) for t in self.times] for s
               in self.times]
        Z = np.random.normal(0, 1, self.Tn)
        C = np.linalg.cholesky(cov)
        return C.dot(Z)


if __name__ == '__main__':
    # Set seed
    random.seed(2023)

    # Simulate paths
    T = 1
    Tn = 1000
    times = [T/Tn * n for n in range(1,Tn+1)]


    W_0_1 = fBm(0.1, T, Tn).simulate()
    W_0_5 = fBm(0.5, T, Tn).simulate()
    W_0_9 = fBm(0.9, T, Tn).simulate()

    # Plot paths
    fig, ((ax1),(ax2),(ax3)) = plt.subplots(ncols=3, figsize=(15,4))
    ax1.plot(times,W_0_1)
    ax1.set_title('H=0.1')
    ax2.plot(times,W_0_5)
    ax2.set_title('H=0.5')
    ax3.plot(times,W_0_9)
    ax3.set_title('H=0.9')
    # fig.subplots_adjust(hspace=0.8)
    plt.savefig('paths.png')
    plt.close()


    # Density plot
    W_df = pd.DataFrame(W_0_1)
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(15,10))
    ax1.hist(np.log(abs(W_df)).diff(1), bins = 50, density=True)
    ax1.set_title('$\Delta=1$')
    ax2.hist(np.log(abs(W_df)).diff(5), bins = 50, density=True)
    ax2.set_title('$\Delta=5$')
    ax3.hist(np.log(abs(W_df)).diff(25), bins = 50, density=True)
    ax3.set_title('$\Delta=25$')
    ax4.hist(np.log(abs(W_df)).diff(125), bins = 50, density=True)
    ax4.set_title('$\Delta=125$')
    fig.subplots_adjust(hspace=0.3)
    plt.savefig('density_fBm.png')
    plt.close()

    fig, ax = plt.subplots(2, 2, figsize = (12,8))
    ax = ax.flatten()
    sns.histplot(np.log(abs(W_df)).diff(1), ax=ax[0], kde=True, legend=False, stat = 'probability')
    ax[0].set_title('$\Delta=1$')
    sns.histplot(np.log(abs(W_df)).diff(5), ax=ax[1], kde=True, legend=False, stat = 'probability')
    ax[1].set_title('$\Delta=5$')
    sns.histplot(np.log(abs(W_df)).diff(25), ax=ax[2], kde=True, legend=False, stat = 'probability')
    ax[2].set_title('$\Delta=25$')
    sns.histplot(np.log(abs(W_df)).diff(125), ax=ax[3], kde=True, legend=False, stat = 'probability')
    ax[3].set_title('$\Delta=125$')
    fig.subplots_adjust(hspace=0.3)
    plt.savefig('density_fBm.png')
    plt.close()

    # QQ plot
    diffs = [1,5,25,125]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    ax = axes.flatten()
    for i in range(4):
        stats.probplot((np.log(abs(W_df)).diff(diffs[i])[diffs[i]:]).transpose().to_numpy()[0], dist='norm', plot=ax[i])
        ax[i].set_title('$\Delta=$' + str(diffs[i]))
        ax[i].get_lines()[0].set_markersize(2)
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.3)
    plt.savefig('QQ_fBm.png')
    plt.close()

