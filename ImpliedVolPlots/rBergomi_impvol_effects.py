# Imports
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from Black_Scholes_methods import implied_volatility
from rModels.rBergomi import rBergomi


# Define parameters
S0 = 100
xi0 = 0.04
V0 = xi0
rho = -0.9
eta = 1.9
H = 0.2
T = 2
Tn = 400
M = 100000

random.seed(2023)

k = np.arange(80,120)

# Varying H
H_para = [0.01, 0.1, 0.2, 0.3, 0.4, 0.49]

dict_impvol_H = dict.fromkeys(H_para)
for h in H_para:
    print(h)
    rB = rBergomi(S0,V0, rho, eta, xi0, h, T, Tn, M)
    rB.covMatrix()
    S, V = rB.mc_simulate()

    df_S = pd.DataFrame(S)

    df_call = pd.DataFrame(range(rB.Tn), columns=['index'])
    for K in k:
        df_call[K] = df_call.apply(lambda x: np.mean(np.maximum(df_S[x['index']] - K, 0)), axis=1)
    df_call['T'] = rB.times

    df_impvol = pd.DataFrame(range(rB.Tn), columns=['index'])
    for K in k:
        df_impvol[K] = df_call.apply(lambda x: implied_volatility(100, K, x['T'], 0, 0, x[K]), axis=1)
    df_impvol.drop('index', axis=1, inplace=True)
    headers = np.log(df_impvol.columns / rB.S0)
    df_impvol.columns = headers

    dict_impvol_H[h] = df_impvol

# Plots implied volatilities for H varying
index = [19, 79, 185, 299, 386]
T_label = np.array(rB.times)[index].round(2)
T_label_str = ["T=" + str(t) for t in T_label]
fig, ax = plt.subplots(2,3, figsize=(10,7), layout='tight')
ax = ax.flatten()
for i in range(6):
    ax[i].plot(dict_impvol_H[H_para[i]].iloc[index].transpose().loc[np.log(85/rB.S0):np.log(115/rB.S0)])
    ax[i].legend(T_label_str)
    ax[i].set_title('H=' + str(H_para[i]))
    ax[i].set_xlabel('logmoneyness')
    ax[i].set_ylabel(r'$\sigma_{BS}$')
    # ax[i].set_ylim([0.09, 0.23])
fig.subplots_adjust(hspace=0.35)
fig.subplots_adjust(wspace=0.35)
plt.savefig(r'Output/Bergomi_H_effect.png')
plt.close()

# Varying the rest
# rho
rho_para = [-0.9, -0.5, -0.1, 0.2, 0.5]
dict_impvol_rho = dict.fromkeys(rho_para)
for r in rho_para:
    print(r)
    rB = rBergomi(S0,V0, r, eta, xi0, H, T, Tn, M)
    rB.covMatrix()
    S, V = rB.mc_simulate()

    df_S = pd.DataFrame(S)

    df_call = pd.DataFrame(range(rB.Tn), columns=['index'])
    for K in k:
        df_call[K] = df_call.apply(lambda x: np.mean(np.maximum(df_S[x['index']] - K, 0)), axis=1)
    df_call['T'] = rB.times

    df_impvol = pd.DataFrame(range(rB.Tn), columns=['index'])
    for K in k:
        df_impvol[K] = df_call.apply(lambda x: implied_volatility(100, K, x['T'], 0, 0, x[K]), axis=1)
    df_impvol.drop('index', axis=1, inplace=True)
    headers = np.log(df_impvol.columns / rB.S0)
    df_impvol.columns = headers
    dict_impvol_rho[r] = df_impvol

index = [19, 79, 185, 299, 386]
T_label = np.array(rB.times)[index].round(3)

# Plot implied volatilities for rho varying
fig, ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.flatten()
for i in range(4):
    if i == 0:
        strikes = np.arange(92,110)
        strikes = np.log(strikes/rB.S0)
    else:
        strikes = np.log(k/rB.S0)
    for r in rho_para:
        ax[i].plot(dict_impvol_rho[r][strikes].iloc[index[i]], label=r'$\rho=$' + str(r))
    ax[i].set_title('T='+str(T_label[i]))
    ax[i].set_xlabel('logmoneyness')
    ax[i].set_ylabel(r'$\sigma_{BS}$')
    ax[i].legend()
fig.subplots_adjust(hspace=0.35)
fig.subplots_adjust(wspace=0.35)
plt.savefig(r'Output/Bergomi_rho_effect.png')
plt.close()

# Varying eta
eta_para = [1.1, 1.5, 1.9, 2.4, 3, 3.5]
dict_impvol_eta = dict.fromkeys(eta_para)
for e in eta_para:
    print(e)
    rB = rBergomi(S0, V0, rho, e, xi0, H, T, Tn, M)
    rB.covMatrix()
    S, V = rB.mc_simulate()

    df_S = pd.DataFrame(S)

    df_call = pd.DataFrame(range(rB.Tn), columns=['index'])
    for K in k:
        df_call[K] = df_call.apply(lambda x: np.mean(np.maximum(df_S[x['index']] - K, 0)), axis=1)
    df_call['T'] = rB.times

    df_impvol = pd.DataFrame(range(rB.Tn), columns=['index'])
    for K in k:
        df_impvol[K] = df_call.apply(lambda x: implied_volatility(100, K, x['T'], 0, 0, x[K]), axis=1)
    df_impvol.drop('index', axis=1, inplace=True)
    headers = np.log(df_impvol.columns / rB.S0)
    df_impvol.columns = headers
    dict_impvol_eta[e] = df_impvol

# Plot implied volatilities for varying eta
fig, ax = plt.subplots(2,2,figsize=(8,8), layout='tight')
ax = ax.flatten()
for i in range(4):
    if i == 0:
        strikes = np.arange(80,115)
        strikes = np.log(strikes/rB.S0)
    else:
        strikes = np.log(k/rB.S0)
    for e in eta_para:
        ax[i].plot(dict_impvol_eta[e][strikes].iloc[index[i]], label=r'$\eta=$' + str(e))
    ax[i].set_title('T='+str(T_label[i]))
    ax[i].set_xlabel('logmoneyness')
    ax[i].set_ylabel(r'$\sigma_{BS}$')
    ax[i].legend()
fig.subplots_adjust(hspace=0.35)
fig.subplots_adjust(wspace=0.35)
plt.savefig(r'Output/Bergomi_eta_effect.png')
plt.close()


# ATM skew
import scipy
H = 0.1
Tn = 1000
T = 1
rB = rBergomi(S0, V0, rho, eta, xi0, H, T, Tn, M)
rB.covMatrix()

S, V = rB.mc_simulate()
df_S = pd.DataFrame(S)

k = rB.S0 * np.exp([-0.01, -0.001, -0.0001, -0.00001, 0.00001, 0.0001, 0.001, 0.01])

# We assume q=r=0
q = 0
r = 0

# Calculate call prices
df_call = pd.DataFrame(range(rB.Tn + 1), columns=['index'])
for K in k:
    df_call[K] = df_call.apply(lambda x: np.mean(np.maximum(df_S[x['index']] - K, 0)), axis=1)
df_call['T'] = np.append(0, rB.times)

df_call['r'] = r

# Calculate implied volatility
df_impvol = pd.DataFrame(range(rB.Tn + 1), columns=['index'])
for K in k:
    df_impvol[K] = df_call.apply(lambda x: implied_volatility(rB.S0, K, x['T'], r, q, x[K]), axis=1)
df_impvol.drop('index', axis=1, inplace=True)

ATM = np.zeros(rB.Tn)
k1 = k[4]
k2 = k[-5]
divisor = np.log(k2/ 100) - np.log(k1 / 100)
for i in range(rB.Tn):
    iv_l = df_impvol.iloc[i][k1]
    iv_h = df_impvol.iloc[i][k2]
    ATM[i] = (iv_h - iv_l) / divisor

b, a = scipy.stats.linregress(np.log(rB.times[7:]), np.log(abs(ATM[7:])))[:2]
print('t^' + str(b))
power_law_fit = lambda x: np.exp(a) * x ** (b)

fig, ax = plt.subplots(figsize=(8,6))
x = rB.times[7:]
text = r'${}\cdot t^{{ {}{:.3f} }}$'.format(round(np.exp(a), 3), '+' if np.sign(b) > 0 else '-', abs(b))
ax.plot(rB.times[7:], abs(ATM[7:]), '.', label='Simulated values', color='red')
ax.plot(x, power_law_fit(x), label='Power law fit: ' + text, color='cornflowerblue')
ax.set_ylabel(r'$|\psi(T)|$')
ax.set_xlabel('T')
ax.legend()
plt.savefig(r'Output/ATM_Bergomi.png')
plt.close()



