# Imports
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from Black_Scholes_methods import implied_volatility
from rModels.rHeston import rHeston


# Define parameters
N = 5
T = 2
Tn = 400  # 100
Lambda = 0.3
rho = -0.7
nu = 0.3
H = 0.1
V0 = 0.02
theta = 0.02
S0 = 100
M = 100000

delta = 1/500
n = 500 # 500
gamma_0 = np.array([1, 3, 10, 40, 100])

random.seed(2023)

k = np.arange(80,120)

# Varying H
H_para = [0.01, 0.1, 0.2, 0.3, 0.4, 0.49]

dict_impvol_H = dict.fromkeys(H_para)
for h in H_para:
    print(h)
    rH_l2optim = rHeston(S0, V0, Lambda, theta, nu, rho, T, Tn, N, M, h)
    c_l2optim, gamma_l2optim = rH_l2optim.optim_l2(delta, n, gamma_0)
    S, V = rH_l2optim.MC_simulate(c_l2optim, gamma_l2optim)

    df_S = pd.DataFrame(S)

    df_call = pd.DataFrame(range(rH_l2optim.Tn), columns=['index'])
    for K in k:
        df_call[K] = df_call.apply(lambda x: np.mean(np.maximum(df_S[x['index']] - K, 0)), axis=1)
    df_call['T'] = rH_l2optim.times

    df_impvol = pd.DataFrame(range(rH_l2optim.Tn), columns=['index'])
    for K in k:
        df_impvol[K] = df_call.apply(lambda x: implied_volatility(100, K, x['T'], 0, 0, x[K]), axis=1)
    df_impvol.drop('index', axis=1, inplace=True)
    headers = np.log(df_impvol.columns / rH_l2optim.S0)
    df_impvol.columns = headers

    dict_impvol_H[h] = df_impvol

# Plot implied volatilities varying H
index = [19, 79, 185, 299, 386]
T_label = np.array(rH_l2optim.times)[index].round(2)
T_label_str = ["T=" + str(t) for t in T_label]
fig, ax = plt.subplots(2,3, figsize=(10,7))
ax = ax.flatten()
for i in range(6):
    ax[i].plot(dict_impvol_H[H_para[i]].iloc[index].transpose().loc[np.log(85/rH_l2optim.S0):np.log(115/rH_l2optim.S0)])
    ax[i].legend(T_label_str)
    ax[i].set_title('H=' + str(H_para[i]))
    ax[i].set_xlabel('logmoneyness')
    ax[i].set_ylabel(r'$\sigma_{BS}$')
fig.subplots_adjust(hspace=0.35)
fig.subplots_adjust(wspace=0.35)
plt.savefig(r'Output/Heston_H_effect.png')
plt.close()

# Varying the rest
# Lambda
Lambda_para = [0.1, 0.3, 0.5, 1, 1.5]
dict_impvol_l = dict.fromkeys(Lambda_para)
for l in Lambda_para:
    print(l)
    rH_l2optim = rHeston(S0, V0, l, theta, nu, rho, T, Tn, N, M, 0.1)
    c_l2optim, gamma_l2optim = rH_l2optim.optim_l2(delta, n, gamma_0)
    S, V = rH_l2optim.MC_simulate(c_l2optim, gamma_l2optim)

    df_S = pd.DataFrame(S)

    df_call = pd.DataFrame(range(rH_l2optim.Tn), columns=['index'])
    for K in k:
        df_call[K] = df_call.apply(lambda x: np.mean(np.maximum(df_S[x['index']] - K, 0)), axis=1)
    df_call['T'] = rH_l2optim.times

    df_impvol = pd.DataFrame(range(rH_l2optim.Tn), columns=['index'])
    for K in k:
        df_impvol[K] = df_call.apply(lambda x: implied_volatility(100, K, x['T'], 0, 0, x[K]), axis=1)
    df_impvol.drop('index', axis=1, inplace=True)
    headers = np.log(df_impvol.columns / rH_l2optim.S0)
    df_impvol.columns = headers
    dict_impvol_l[l] = df_impvol

# Plot implied volatilities varying lambda
index = [19, 79, 185, 299, 386]
T_label = np.array(rH_l2optim.times)[index].round(2)

fig, ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.flatten()
for i in range(4):
    if i == 0:
        strikes = np.arange(92,110)
        strikes = np.log(strikes/rH_l2optim.S0)
    else:
        strikes = np.log(k/rH_l2optim.S0)
    for l in Lambda_para:
        ax[i].plot(dict_impvol_l[l][strikes].iloc[index[i]], label=r'$\lambda=$' + str(l))
    ax[i].set_title('T='+str(T_label[i]))
    ax[i].set_xlabel('logmoneyness')
    ax[i].set_ylabel(r'$\sigma_{BS}$')
    ax[i].legend()
fig.subplots_adjust(hspace=0.35)
fig.subplots_adjust(wspace=0.35)
plt.savefig(r'Output/Heston_lambda_effect.png')
plt.close()

# Varying nu
rH_l2optim.Lambda = 0.3
nu_para = Lambda_para
dict_impvol_nu = dict.fromkeys(nu_para)
for nu_p in nu_para:
    print(nu_p)
    rH_l2optim = rHeston(S0, V0, 0.3, theta, nu_p, rho, T, Tn, N, M, 0.1)
    c_l2optim, gamma_l2optim = rH_l2optim.optim_l2(delta, n, gamma_0)
    S, V = rH_l2optim.MC_simulate(c_l2optim, gamma_l2optim)

    df_S = pd.DataFrame(S)

    df_call = pd.DataFrame(range(rH_l2optim.Tn), columns=['index'])
    for K in k:
        df_call[K] = df_call.apply(lambda x: np.mean(np.maximum(df_S[x['index']] - K, 0)), axis=1)
    df_call['T'] = rH_l2optim.times

    df_impvol = pd.DataFrame(range(rH_l2optim.Tn), columns=['index'])
    for K in k:
        df_impvol[K] = df_call.apply(lambda x: implied_volatility(100, K, x['T'], 0, 0, x[K]), axis=1)
    df_impvol.drop('index', axis=1, inplace=True)
    headers = np.log(df_impvol.columns / rH_l2optim.S0)
    df_impvol.columns = headers
    dict_impvol_nu[nu_p] = df_impvol

# Plot implied volatilities for varying nu
fig, ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.flatten()
for i in range(4):
    if i == 0:
        strikes = np.arange(92,110)
        strikes = np.log(strikes/rH_l2optim.S0)
    else:
        strikes = np.log(k/rH_l2optim.S0)
    for nu_p in nu_para:
        ax[i].plot(dict_impvol_nu[nu_p][strikes].iloc[index[i]], label=r'$\nu=$' + str(nu_p))
    ax[i].set_title('T='+str(T_label[i]))
    ax[i].set_xlabel('logmoneyness')
    ax[i].set_ylabel(r'$\sigma_{BS}$')
    ax[i].legend()
fig.subplots_adjust(hspace=0.35)
fig.subplots_adjust(wspace=0.35)
plt.savefig(r'Output/Heston_nu_effect.png')
plt.close()

# Varying rho
rho_para = [-0.9, -0.5, -0.1, 0.5, 0.9]
dict_impvol_rho = dict.fromkeys(rho_para)
for r in rho_para:
    print(r)
    rH_l2optim = rHeston(S0, V0, 0.3, theta, 0.3, r, T, Tn, N, M, 0.1)
    c_l2optim, gamma_l2optim = rH_l2optim.optim_l2(delta, n, gamma_0)
    S, V = rH_l2optim.MC_simulate(c_l2optim, gamma_l2optim)

    df_S = pd.DataFrame(S)

    df_call = pd.DataFrame(range(rH_l2optim.Tn), columns=['index'])
    for K in k:
        df_call[K] = df_call.apply(lambda x: np.mean(np.maximum(df_S[x['index']] - K, 0)), axis=1)
    df_call['T'] = rH_l2optim.times

    df_impvol = pd.DataFrame(range(rH_l2optim.Tn), columns=['index'])
    for K in k:
        df_impvol[K] = df_call.apply(lambda x: implied_volatility(100, K, x['T'], 0, 0, x[K]), axis=1)
    df_impvol.drop('index', axis=1, inplace=True)
    headers = np.log(df_impvol.columns / rH_l2optim.S0)
    df_impvol.columns = headers
    dict_impvol_rho[r] = df_impvol

# Plot implied volatilities for rho varying
fig, ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.flatten()
for i in range(4):
    if i == 0:
        strikes = np.arange(92,110)
        strikes = np.log(strikes/rH_l2optim.S0)
    else:
        strikes = np.log(k/rH_l2optim.S0)
    for r in rho_para:
        ax[i].plot(dict_impvol_rho[r][strikes].iloc[index[i]], label=r'$\rho=$' + str(r))
    ax[i].set_title('T='+str(T_label[i]))
    ax[i].set_xlabel('logmoneyness')
    ax[i].set_ylabel(r'$\sigma_{BS}$')
    ax[i].legend()
fig.subplots_adjust(hspace=0.35)
fig.subplots_adjust(wspace=0.35)
plt.savefig(r'Output/Heston_rho_effect.png')
plt.close()

# ATM skew
import scipy
N = 5
gamma_0 = np.array([1,3,5,10,100])
n = 1000
delta = 1/1000
Tn = 1000
T = 1
rH = rHeston(S0, V0, 0.3, theta, nu, rho, T, Tn, N, M, 0.15)
c, gamma = rH.optim_l2(delta, n, gamma_0)

S, V = rH.MC_simulate(c, gamma)
df_S = pd.DataFrame(S)

# We assume q=r=0
q = 0
r = 0

k = rH.S0 * np.exp([-0.01, -0.001, -0.0001, -0.00001, 0.00001, 0.0001, 0.001, 0.01])

# Calculate call prices
df_call = pd.DataFrame(range(rH.Tn + 1), columns=['index'])
for K in k:
    df_call[K] = df_call.apply(lambda x: np.mean(np.maximum(df_S[x['index']] - K, 0)), axis=1)
df_call['T'] = np.append(0, rH.times)

df_call['r'] = r

# Calculate implied volatilities
df_impvol = pd.DataFrame(range(rH.Tn + 1), columns=['index'])
for K in k:
    df_impvol[K] = df_call.apply(lambda x: implied_volatility(rH.S0, K, x['T'], x['r'], q, x[K]), axis=1)
df_impvol.drop('index', axis=1, inplace=True)

# Calculate ATM skew
ATM = np.zeros(rH.Tn)
k1 = k[4]
k2 = k[-5]
divisor = np.log(k2 / 100) - np.log(k1 / 100)
for i in range(rH.Tn):
    iv_l = df_impvol[k1].iloc[i+1]
    iv_h = df_impvol[k2].iloc[i+1]
    ATM[i] = (iv_h - iv_l) / divisor

b, a = scipy.stats.linregress(np.log(rH.times[7:]), np.log(abs(ATM[7:])))[:2]
print('t^' + str(b))
power_law_fit = lambda x: np.exp(a) * x ** (b)

# Plot ATM skew
fig, ax = plt.subplots(figsize=(8,6))
x = rH.times[7:]
text = r'${}\cdot t^{{ {}{:.3f} }}$'.format(round(np.exp(a), 3), '+' if np.sign(b) > 0 else '-', abs(b))
ax.plot(rH.times[7:], abs(ATM[7:]), '.', label='Simulated values', color='red')
ax.plot(x, power_law_fit(x), label='Power law fit: ' + text, color='cornflowerblue')
ax.set_ylabel(r'$|\psi(T)|$')
ax.set_xlabel('T')
ax.legend()
plt.savefig(r'Output/ATM_heston.png')
plt.close()
