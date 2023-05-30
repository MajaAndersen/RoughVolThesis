# imports
import datetime

import matplotlib.pyplot as plt
import numpy as np
import random

import pandas as pd
import scipy
import seaborn as sns

from rModels.rBergomi import rBergomi
from Black_Scholes_methods import implied_volatility

# Set parameters
S0 = 100
xi0 = 0.04
V0 = xi0
rho = -0.9
eta = 1.9
H = 0.1
T = 1
Tn = 1000
M = 50000

random.seed(2023)

# Create the model object
rB = rBergomi(S0, V0, rho, eta, xi0, H, T, Tn, M)
rB.covMatrix()

# Simulate from the two methods
# Rough
dt1 = datetime.datetime.now()
rB.covMatrix()
S, V = rB.mc_simulate()
dt2 = datetime.datetime.now()


# Multifactor
# We want to test 3 different numbers of factors
N = [1,3,5,10]
gamma_0 = {1: [1],
           3: [1,10,100],
           4: [1, 10, 30, 100],
           5: [1,3,10, 30,100],
           10: [1,2,3,10,20,40,60,80,100,130]}
delta = 1/(rB.Tn*10)
n = rB.Tn
S_mf = dict.fromkeys(N)
V_mf = dict.fromkeys(N)
dts = []
for Ni in N:
    print(Ni)
    dt1 = datetime.datetime.now()
    s, v = rB.simulate_multifactor(gamma_0[Ni], delta, n)
    dt2 = datetime.datetime.now()
    dts.append((dt2-dt1).total_seconds())
    S_mf[Ni] = s
    V_mf[Ni] = v

# Plot density of asset prices
fig, ax = plt.subplots(figsize=(8,6))
sns.kdeplot(S.flatten(), label='Exact simulation', color='black')
for Ni in N:
    sns.kdeplot(S_mf[Ni].flatten(), label='N='+str(Ni))
plt.legend()
plt.savefig(r'Output/Bergomi_multifactor_S.png')
plt.close()


# Plot density of log volatility increments (we have chosen a difference of 10 in increments)
log_V = np.log(np.sqrt(V))
log_V_5 = []
for v in log_V:
    log_V_5.append(v[10:] - v[:-10])
log_V_5 = log_V[:,10:] - log_V[:,:-10]
log_V_5 = np.array(log_V_5)

sns.kdeplot(log_V_5.flatten(), label='rough bergomi', color='black')

fig, ax = plt.subplots(figsize=(8,6))
sns.kdeplot(log_V_5.flatten(), label='rough bergomi', color='black')
for Ni in N:
    sns.kdeplot((np.log(np.sqrt(V_mf[Ni]))[:,10:]-np.log(np.sqrt(V_mf[Ni]))[:,:-10]).flatten(), label='N='+str(Ni))
plt.legend()
plt.show()
plt.savefig(r'Output/Bergomi_multifactor_V.png')
plt.close()

# Calculate call prices and implied volatilities
strikes = np.arange(99.9, 100.1, 1/100)

dict_impvol = dict.fromkeys(N)
for Ni in N:
    df_S = pd.DataFrame(S_mf[Ni])
    df_call = pd.DataFrame(range(rB.Tn + 1), columns=['index'])
    for K in strikes:
        df_call[K] = df_call.apply(lambda x: 1/rB.M * (np.maximum(df_S[int(x['index'])] - K, 0)).sum(), axis=1)
    df_call['T'] = np.append(0, rB.times)

    df_impvol = pd.DataFrame(range(rB.Tn + 1), columns=['index'])
    for K in strikes:
        df_impvol[K] = df_call.apply(lambda x: implied_volatility(rB.S0, K, x['T'], 0, 0, x[K]), axis=1)
    df_impvol.drop('index', axis=1, inplace=True)

    dict_impvol[Ni] = df_impvol

# ATM skew
ATM_N = np.zeros(rB.Tn * len(N)).reshape((rB.Tn,len(N)))
for j in range(len(N)):
    Ni = N[j]
    ATM = np.zeros(rB.Tn)
    k1 = strikes[9]
    k2 = strikes[11]
    divisor = np.log(k2 / 100) - np.log(k1 / 100)
    for i in range(rB.Tn):
        iv_l = dict_impvol[Ni][k1].iloc[i + 1]
        iv_h = dict_impvol[Ni][k2].iloc[i + 1]
        ATM[i] = (iv_h - iv_l) / divisor
    ATM_N[:,j] = ATM

# Do the same for exact simulation
# Call prices and implied volatilities
df_S = pd.DataFrame(S)
df_call = pd.DataFrame(range(rB.Tn + 1), columns=['index'])
for K in strikes:
    df_call[K] = df_call.apply(lambda x: 1 / rB.M * (np.maximum(df_S[int(x['index'])] - K, 0)).sum(), axis=1)
df_call['T'] = np.append(0, rB.times)

df_impvol = pd.DataFrame(range(rB.Tn + 1), columns=['index'])
for K in strikes:
    df_impvol[K] = df_call.apply(lambda x: implied_volatility(rB.S0, K, x['T'], 0, 0, x[K]), axis=1)
df_impvol.drop('index', axis=1, inplace=True)

# ATM skew
ATM = np.zeros(rB.Tn)
# k1 = 99
# k2 = 101
divisor = np.log(k2 / 100) - np.log(k1 / 100)
for i in range(rB.Tn):
    iv_l = df_impvol[k1].iloc[i + 1]
    iv_h = df_impvol[k2].iloc[i + 1]
    ATM[i] = (iv_h - iv_l) / divisor

# Create plots
fig, ax = plt.subplots(figsize=(10,8))
x = rB.times[7:]

b, a = scipy.stats.linregress(np.log(x), np.log(abs(ATM[7:])))[:2]
print('t^' + str(b))
power_law_fit = lambda x: np.exp(a) * x ** (b)

ax.plot(x, abs(ATM[7:]), '.', label='Exact', color='black')
ax.plot(x, power_law_fit(x), label='Exact', color='black')

for i in range(len(N)):
    b, a = scipy.stats.linregress(np.log(x), np.log(abs(ATM_N[7:,i])))[:2]
    # print('t^' + str(b))
    power_law_fit = lambda x: np.exp(a) * x ** (b)

    ax.scatter(x, abs(ATM_N[7:,i]), label='N=' + str(N[i]), s=3)

ax.legend()
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$|\psi(T)|$')
plt.savefig(r'Output/ATM_skew_bergomi_multifactor.png')
plt.close()
