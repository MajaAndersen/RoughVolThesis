# Imports
import datetime
import random

import numpy.linalg
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import datetime as dt

#from line_profiler import LineProfiler
from Black_Scholes_methods import implied_volatility

# Create class rBergomi
class rBergomi:
    def __init__(self, S0, V0, rho, eta, xi0, H, T, Tn, M):
        self.S0 = S0
        self.V0 = V0
        self.rho = rho
        self.eta = eta
        self.xi0 = xi0
        self.H = H
        self.gamma = 1/2 - self.H
        self.alpha = self.H - 1/2
        self.T = T
        self.Tn = Tn
        self.dt = T / Tn
        self.times = np.arange(start=self.dt, stop=self.T + self.dt, step=self.dt) #[self.dt * n for n in range(1,self.Tn+1)]
        self.M = M

    def G(self, x):
        F = scipy.special.hyp2f1(self.gamma, 1, 2-self.gamma, x)
        G = (1 - 2*self.gamma)/(1 - self.gamma) * x**self.gamma * F
        return G

    def covMatrix(self):
        sigma11 = [[min(u, v) for u in self.times] for v in self.times]
        sigma12 = [[self.rho * np.sqrt(2 * self.H) / (self.H + 0.5) * (v ** (self.H + 0.5) - (v - min(u, v)) ** (self.H + 0.5)) for u in self.times]
                   for v in self.times]
        sigma22 = [[min(u, v) ** (2 * self.H) * self.G(min(u, v) / max(u, v)) for u in self.times] for v in self.times]

        sigma1 = np.block([np.array(sigma11), np.transpose(sigma12)])
        sigma2 = np.block([np.array(sigma12), np.array(sigma22)])

        self.cov = np.block([[sigma1],[sigma2]])
        try:
            self.C = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            self.C = np.linalg.cholesky(self.cov + np.diag([1e-6]*self.Tn*2))

    def is_symmetrix(self):
        if not np.allclose(self.cov, np.transpose(self.cov)):
            print('Covariance matrix not symmetric')
        else:
            print('Covariance matrix is symmetric :)')

    def simulate(self):
        # Create lists for vol and underlying price paths with inital values
        V = np.zeros(self.Tn + 1)
        S = np.zeros(self.Tn + 1)
        V[0] = self.V0
        S[0] = self.S0

        # Create brownian motion terms
        Z = np.random.normal(0, 1, 2 * self.Tn)
        C = self.C
        if np.shape(C)[0] != np.shape(Z)[0]:
            print('Spøjst')
        W = C.dot(Z)
        W0 = W[-self.Tn:]
        W2 = W[:self.Tn]
        W2_shifted = np.insert(np.diff(W2, 1), 0, 0)


        # Calculate values for V and S for each timepoint
        V[1:] = self.xi0 * np.exp(self.eta * W0 - self.eta ** 2 / 2 * (self.times - 0) ** (2 * self.H))
        for i in range(self.Tn):
            S[i+1] = S[i] + np.sqrt(V[i]) * S[i] * W2_shifted[i]

        return S, V

    def mc_simulate(self):
        vols = np.zeros((self.M, self.Tn + 1))
        prices = np.zeros((self.M, self.Tn + 1))

        for i in range(self.M):
            s, v = self.simulate()
            prices[i, :] = s[:]
            vols[i, :] = v

        return prices, vols

    # Functionalities needed for multifactor simulations
    def K(self, t):
        c_alpha = (scipy.special.gamma(1 + self.alpha))**(-1)
        return c_alpha * t ** self.alpha

    def A(self, gamma, times):
        A_return = np.zeros((len(times), self.N))
        for i in range(self.Tn):
            for j in range(self.N):
                A_return[i, j] = np.exp(-gamma[j] * times[i])
        return A_return

    def K_hat(self, times, c, gamma):
        sum = np.zeros(len(times))
        for i in range(len(times)):
            for j in range(self.N):
                sum[i] += np.exp(-gamma[j] * times[i]) * c[j]
        return sum

    def optim_l2(self, delta, n, gamma_0):
        dt = (self.T - delta) / n
        times = np.arange(delta, self.T, dt)
        y = np.array([self.K(t) for t in times])

        def gamma_(epsilon):
            return np.cumsum(epsilon)

        def c_(gamma):
            return np.matmul(np.linalg.pinv(self.A(gamma, times)), y)

        epsilon_0 = np.hstack((gamma_0[0], np.diff(gamma_0))).reshape((1,self.N))
        def obj_fun(epsilon):
            return ((y - self.K_hat(times, c_(gamma_(epsilon)), gamma_(epsilon)))**2).sum()

        bounds = tuple([(0,None) for x in gamma_0])
        optim = scipy.optimize.minimize(obj_fun, epsilon_0,bounds=bounds)

        epsilon_result = optim['x']
        gamma_result = gamma_(epsilon_result)
        c_result = c_(gamma_result)
        c_result = c_result * scipy.special.gamma(self.H + 1/2)

        return c_result, gamma_result

    def simulate_multifactor(self, gamma_0, delta, n):
        self.N = len(gamma_0)

        c_result, gamma_result = self.optim_l2(delta, n, gamma_0)

        # Do actual simulation
        vols = np.zeros((self.M, self.Tn + 1))
        prices = np.zeros((self.M, self.Tn + 1))

        for m in range(self.M):

            V = np.zeros(self.Tn + 1)
            V[0] = self.V0
            S = np.zeros(self.Tn + 1)
            S[0] = self.S0
            U = np.zeros(self.N)

            W = np.random.normal(0, 1, self.Tn) * np.sqrt(self.dt)
            W_perp = np.random.normal(0, 1, self.Tn) * np.sqrt(self.dt)

            def U_cov(t):
                Ucov = np.zeros(self.N ** 2).reshape((self.N, self.N))
                for i in range(self.N):
                    for j in range(self.N):
                        gamma_sum = gamma_result[i] + gamma_result[j]
                        Ucov[i][j] = 1 / gamma_sum * (1 - np.exp(-gamma_sum * t))
                return Ucov

            U_saved = np.zeros(self.N * (self.Tn + 1)).reshape((self.Tn + 1, self.N))
            U_saved[0] = U


            for i in range(self.Tn):
                var_factor = U_cov(self.dt * (i + 1))
                dW = W[i]
                dB = self.rho * W[i] + np.sqrt(1 - self.rho ** 2) * W_perp[i]
                U = 1 / (1 + gamma_result * self.dt) * (U + dW)
                # C_U[i] = np.dot(c_result, U)
                U_saved[i + 1] = U

                c_factor = np.sqrt(2 * self.H) * c_result

                var_term = c_factor @ var_factor @ c_factor
                V[i + 1] = self.xi0 * np.exp(
                    self.eta * np.dot(c_factor, U) - 0.5 * self.eta ** 2 * var_term)
                S[i + 1] = S[i] + np.sqrt(V[i]) * S[i] * dB

            vols[m, :] = V
            prices[m, :] = S

        return prices, vols



    def calibrate(self, df_observed):
        # Check columns are named correctly
        if 'k' not in df_observed.columns or 'imp_vol' not in df_observed.columns or 'TTM' not in df_observed.columns or 'r' not in df_observed.columns or 'q' not in df_observed.columns:
            ValueError('The columns have to include "k", "imp_vol", "TTM", "r" and "q" with that specific spelling')

        # n = len(df_observed)
        # Place observed implied volatilities in array
        df_impvol = pd.pivot_table(df_observed, values='imp_vol', columns='k', index='TTM')

        T = df_observed.TTM.unique()
        k = df_observed.k.unique() * self.S0

        q = df_observed.q.iloc[0]

        random.seed(2023)

        # Initial guess for optimization
        initial_params = [self.H, self.rho, self.eta]

        def obj(params):
            H, rho, eta = params

            # Create array of zeroes to fill with calculated implied volatilities
            imp_vol_hat = np.zeros(len(T) * len(k)).reshape((len(T), len(k)))


            # Parameters need to be replaced be the once we test for
            self.H = H
            self.rho = rho
            self.eta = eta

            print(H)
            print(rho)
            print(eta)

            # Create covariance
            try:
                self.covMatrix()
            except np.linalg.LinAlgError:
                return 1e6

            # Simulate paths
            S, V = self.mc_simulate()
            # Calculate implied volatility for each set of time to maturity and strike
            for i in range(len(T)):
                t = T[i]
                r = df_observed[df_observed.TTM == t].r.iloc[0]
                # Get time index for simulations
                # t_i = round(t/self.dt) - 1
                for j in range(len(k)):
                    K = k[j]
                    C_hat = np.mean(np.maximum(S[:, -i] - K, 0))
                    imp_vol_hat[i,j] = implied_volatility(self.S0, K, t, r, q, C_hat)
                    # imp_vol_hat[i,j] = implied_volatility_call(C_hat, self.S0, K, t, r, q)
            df_impvol_hat = pd.DataFrame(imp_vol_hat, columns=k/self.S0, index=T)

            return ((df_impvol_hat - df_impvol)**2).sum(skipna=True).sum()

        # Bound on H, rho and eta
        bounds = ((0, 0.5), (-1, 1), (1, 4))

        dt1 = datetime.datetime.now()
        res = scipy.optimize.minimize(obj, initial_params,method='L-BFGS-B', bounds=bounds, options={'maxiter':500, 'disp': True})
        dt2 = datetime.datetime.now()
        print(dt2-dt1)

        print(res['message'])
        params = res['x']
        self.H = params[0]
        self.rho = params[1]
        self.eta = params[2]


if __name__ == '__main__':
    # Parameters
    S0 = 100
    xi0 = 0.04
    V0 = xi0
    rho = -0.9
    eta = 1.9
    H = 0.1
    T = 2
    Tn = 400

    random.seed(2023)
    # Create small view of paths
    M = 5
    rB = rBergomi(S0, V0, rho, eta, xi0, H, T, Tn, M)

    rB.covMatrix()
    S, V = rB.mc_simulate()

    # Plot S and V paths
    df_S = pd.DataFrame(S)
    df_V = pd.DataFrame(V)

    fig, ax = plt.subplots(ncols=2, figsize=(12,6))
    ax = ax.flatten()
    ax[0].plot(np.append(0, rB.times), df_S.transpose())
    ax[0].title.set_text('Stock price paths')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('$S_t$')
    ax[1].plot(np.append(0, rB.times), df_V.transpose())
    ax[1].title.set_text('Volatility paths')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('$\sigma_t$')

    plt.savefig('bergomi_paths_5.png')
    plt.close()
    # plt.show()

    # Simulate 10.000 paths to do sanity check E[V]
    M = 100000
    rB.M = M

    rB.covMatrix()
    S, V = rB.mc_simulate()

    df_S = pd.DataFrame(S)
    df_V = pd.DataFrame(V)

    # Sanity check of mean of volatility paths

    timepoints = rB.times
    plt.plot(np.append(0, timepoints), V.mean(axis=0))
    plt.plot(np.append(0, timepoints), V.mean(axis=0) + 1.96 * V.std(axis=0) / np.sqrt(rB.M))
    plt.plot(np.append(0, timepoints), V.mean(axis=0) - 1.96 * V.std(axis=0) / np.sqrt(rB.M))
    plt.axhline(V0,linestyle='--')
    plt.show()

    E_V = df_V.mean(axis=0)

    fig, ax = plt.subplots()
    ax.plot(np.append(0, rB.times), E_V, label='$E(V)$')
    ax.axhline(rB.xi0, color='red', label=r'$\xi_0$')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('E(V)')

    plt.close()
