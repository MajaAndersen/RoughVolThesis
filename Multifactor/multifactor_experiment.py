# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random

import datetime

from rModels.rHeston import rHeston

# Function to find H
def find_H(logV):
    m = np.zeros(shape=(len(x), len(q)))

    for i in range(len(q)):
        m[:,i] = [np.mean(np.abs(pd.Series(logV) - pd.Series(logV).shift(delta))**q[i]) for delta in x]

    model = np.zeros((len(q), 1))
    for i in range(len(q)):
        z = LinearRegression().fit(np.log(x).reshape(-1, 1), np.log(m[:, i].reshape(-1, 1)))
        model[i] = z.coef_

    b, a = np.polyfit(q, model[:, 0], 1)
    return b, a

# Parameters
N = 5
T = 2
Tn = 500  # 100
Lambda = 0.3
rho = -0.7
nu = 0.05
H = 0.1
V0 = 0.1
theta = 0.02
S0 = 100
M = 1000
Tn = 1000

delta = 1/10000
n = 10000
Ns = np.arange(1,7)
gamma_0 = {1: np.array([1]),
           2: np.array([1, 100]),
           3: np.array([1, 10, 100]),
           4: np.array([1, 3, 10, 100]),
           5: np.array([1, 3, 10, 40, 100]),
           6: np.array([1, 3, 10, 40, 60, 100])}
random.seed(2023)

q = [0.5, 1, 1.5, 2, 3]
x = np.arange(1, 50)

# Create model object
estimated_H = []
interception = []
for N in Ns:
    multiFactor = rHeston(S0, V0, Lambda, theta, nu, rho, T, Tn, N, M, H)
    c, gamma = multiFactor.optim_l2(delta, n, gamma_0[N])

    # Simulate
    dt1 = datetime.datetime.now()
    S, V = multiFactor.MC_simulate(c, gamma)
    dt2 = datetime.datetime.now()
    print(dt2-dt1)

    estimation = [find_H(v) for v in np.log(np.sqrt(V))]
    estimated_H.append([e[0] for e in estimation])
    interception.append([e[1] for e in estimation])


fig, ax = plt.subplots(2,3, figsize=(10,7))
ax = ax.flatten()
for i in range(6):
    ax[i].hist(estimated_H[i], bins=50)
    ax[i].set_title('N='+str(i+1))
    ax[i].set_xlabel('H')
fig.subplots_adjust(hspace=0.3)
plt.show()
plt.savefig(r'Output/multifactor_hist.png')
plt.close()
