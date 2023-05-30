# Imports
import datetime
import random

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import datetime as dt

import seaborn as sns

from Black_Scholes_methods import implied_volatility


class rHeston:
    def __init__(self, S0, V0, Lambda, theta, nu, rho, T, Tn, N, M, H):
        self.S0 = S0
        self.V0 = V0  # initial variance
        self.Lambda = Lambda  # mean reversion speed
        self.theta = theta  # mean reversion level
        self.nu = nu  # volatility of variance
        self.rho = rho  # correlation between stock and variance
        self.T = T  # time horizon
        self.Tn = Tn # number of time steps
        self.N = N  # number of factors
        self.M = M  # number of Monte Carlo simulations
        self.H = H  # Hurst exponent
        self.alpha = H - 1/2 # alpha parameter
        self.dt = T / Tn  # time step size
        self.sqrt_dt = np.sqrt(self.dt)
        self.times = [self.dt * n for n in range(1,self.Tn+1)]

    # def g_n(self, t, c, gamma):
    #     return V0 + np.sum([c[i] * self.theta * (1-np.exp(-gamma[i]*t))/gamma[i] for i in range(self.N)])

    def g_n(self, t, c, gamma):
        # return V0 + np.sum(c * self.theta * (1 - np.exp(-gamma * t)) * (1 / gamma))
        return self.V0 + self.Lambda*self.theta * np.sum(c*1/gamma * (1-np.exp(-gamma * t)))

    def simulate(self, c, gamma):
        S = np.zeros(self.Tn + 1)
        S[0] = self.S0
        V = np.zeros(self.Tn + 1)
        V[0] = self.V0
        U = np.zeros(self.N * (self.Tn + 2)).reshape(self.Tn + 2, self.N)
        dt_sqrt = np.sqrt(self.dt)
        dW = np.random.normal(0, 1, self.Tn + 1) * dt_sqrt
        dW2 = np.random.normal(0, 1, self.Tn + 1) * dt_sqrt

        # Remember, update equation looks like; in hopefully obvious notation, see paper for details:
        # g_0 = V_0 + lambda * theta * sum( c_i * int_0^t exp(-gamma_i*(t-s)) ds, i=1,...,n)
        # V = g_0 + c_1 * U_1 + ... + c_n * U_n
        # U = (1 / (1 + gamma * dt)) * (U - lambda * V * dt + nu * sqrt(max(V,0)) * dW

        # Helping varibles to make code faster
        # t_help = np.repeat(np.arange(self.Tn), self.N).reshape(self.Tn, self.N)
        # theta_gamma_t = self.theta * (1 - np.exp(-gamma * t_help)) / gamma
        U_help1 = 1 / (1 + gamma * self.dt)
        # U_help2 = self.Lambda * self.dt + self.nu
        V_sqrt = np.sqrt(max(0, V[0]))
        dB_help = np.sqrt(1 - self.rho ** 2)

        gamma_t_help = [-gamma * t for t in self.times]

        for t in range(self.Tn + 1):
            dWt = dW[t]
            dB = self.rho * dWt + dB_help * dW2[t]
            if t > 0:
                # V[t] = V0 + np.dot(c, self.theta * (1 - np.exp(-gamma * t)) / gamma) + np.dot(c, U)
                # V[t] = V0 + np.dot(c, theta_gamma_t[t]) + np.dot(c, U)
                # V[t] = self.g_n(t * self.dt, c, gamma) + np.dot(c, U[t])
                gn_t = self.V0 + self.Lambda * self.theta * np.sum(c * 1 / gamma * (1 - np.exp(gamma_t_help[t-1])))
                V[t] = gn_t + np.dot(c, U[t])
                V_sqrt = np.sqrt(max(0, V[t]))
                V[t] = V_sqrt**2
                S[t] = (S[t - 1] + S[t - 1] * V_sqrt * dB)
            U[t+1] = U_help1 * (U[t] - self.Lambda * V[t] * self.dt + self.nu * V_sqrt * dWt)
            # V[t] * U_help2 * V_sqrt * dWt
            # U *= U_help1
            # U = np.multiply(U_help1, (U - V[t] * U_help2 * V_sqrt * dWt))
        return V, S

    def MC_simulate(self, c, gamma):
        # Helping parameters defined before to make faster
        U_help = 1 / (1 + gamma * self.dt)
        gamma_t_help = [-gamma * t for t in self.times]
        gn_t = np.zeros(self.Tn)
        for t in range(self.Tn):
            gn_t[t] = self.V0 + self.Lambda * self.theta * np.sum(c * 1 / gamma * (1 - np.exp(gamma_t_help[t])))

        vols = np.zeros((self.M, self.Tn + 1))
        prices = np.zeros((self.M, self.Tn + 1))

        for i in range(self.M):
            S = np.zeros(self.Tn + 1)
            S[0] = self.S0
            V = np.zeros(self.Tn + 1)
            V[0] = self.V0
            U = np.zeros(self.N * (self.Tn + 2)).reshape(self.Tn + 2, self.N)
            dt_sqrt = np.sqrt(self.dt)
            dW = np.random.normal(0, 1, self.Tn + 1) * dt_sqrt
            dW2 = np.random.normal(0, 1, self.Tn + 1) * dt_sqrt

            # Helping parameters needed defined for each simulation
            V_sqrt = np.sqrt(max(0, V[0]))
            dB_help = np.sqrt(1 - self.rho ** 2)

            # Create paths
            for t in range(self.Tn + 1):
                dWt = dW[t]
                dB = self.rho * dWt + dB_help * dW2[t]
                if t > 0:
                    V[t] = gn_t[t-1] + np.dot(c, U[t])
                    V_sqrt = np.sqrt(max(0, V[t]))
                    V[t] = V_sqrt ** 2
                    S[t] = (S[t - 1] + S[t - 1] * V_sqrt * dB)
                U[t + 1] = U_help * (U[t] - self.Lambda * V[t] * self.dt + self.nu * V_sqrt * dWt)

            prices[i, :] = S
            vols[i, :] = V


        return prices, vols

    def set_c_and_gamma_lifting_heston(self, r_N):
        print("m is set to " + str(self.N) + ", and rm is set to " + str(r_N))
        alpha2 = self.H + 1 / 2
        c = [(r_N ** (1 - alpha2) - 1) * r_N ** ((alpha2 - 1) * (1 + self.N / 2)) / (
                    scipy.special.gamma(alpha2) * scipy.special.gamma(2 - alpha2)) * r_N ** ((1 - alpha2) * i) for i in
             range(self.N)]
        gamma = [(1 - alpha2) / (2 - alpha2) * (r_N ** (2 - alpha2) - 1) / (r_N ** (1 - alpha2) - 1) * r_N ** (
                    i - 1 - self.N / 2) for i in range(self.N)]
        return c, gamma

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

        return c_result, gamma_result


if __name__ == '__main__':
    # Define parameters
    N = 100
    T = 1
    Tn = 400
    Lambda = 0.3
    rho = -0.7
    nu = 0.3
    H = 0.1
    V0 = 0.02
    theta = 0.02
    S0 = 100
    M = 10000

    # Create Rough Heston objects
    rH_lifting = rHeston(S0, V0, Lambda, theta, nu, rho, T, Tn, N, M, H)
    rH_l2optim = rHeston(S0, V0, Lambda, theta, nu, rho, T, Tn, 5, M, H)

    # Estimate c and gamma for both methods
    c_l2optim, gamma_l2optim = rH_l2optim.optim_l2(delta=1/100, n=500, gamma_0=np.array([1, 2, 3, 10,100]))
    c_lifting, gamma_lifting = rH_lifting.set_c_and_gamma_lifting_heston(r_N=2.5)

    N_sim = 10000
    s = np.zeros(shape=(N_sim, rH_l2optim.Tn + 1))
    v = np.zeros(shape=(N_sim, rH_l2optim.Tn + 1))
    for i in range(N_sim):
        v[i, :], s[i, :] = rH_l2optim.simulate(c_l2optim, gamma_l2optim)

    rH_l2optim.M = 10
    S, V = rH_l2optim.MC_simulate(c_l2optim, gamma_l2optim)

    # Check the mean of S (should be 1):
    timepoints = rH_l2optim.dt * np.arange(0, rH_l2optim.Tn + 1)
    plt.plot(timepoints, s.mean(axis=0))
    plt.plot(timepoints, s.mean(axis=0) + 1.96 * s.std(axis=0) / np.sqrt(N_sim))
    plt.plot(timepoints, s.mean(axis=0) - 1.96 * s.std(axis=0) / np.sqrt(N_sim))
    plt.show()

    # Check the mean of V:
    plt.plot(timepoints, v.mean(axis=0))
    plt.plot(timepoints, v.mean(axis=0) + 1.96 * V.std(axis=0) / np.sqrt(N_sim))
    plt.plot(timepoints, v.mean(axis=0) - 1.96 * V.std(axis=0) / np.sqrt(N_sim))
    plt.show()

    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax = ax.flatten()
    ax[0].plot(timepoints, s[:10].transpose())
    ax[0].title.set_text('Stock price paths')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel('$S_t$')
    ax[1].plot(timepoints, v[:10].transpose())
    ax[1].title.set_text('Variance paths')
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('$\sigma_t$')
    plt.savefig(r'Output/Heston_paths.png')

    # Lifting Heston kernel approximation
    timepoints = np.linspace(T/500, 1, 1000)

    plt.plot(timepoints, np.array(rH_lifting.K(timepoints)), label='Actual kernel')
    plt.plot(timepoints, np.array(rH_lifting.K_hat(timepoints, c_lifting, gamma_lifting)), '--', label='Lifting Heston kernel')
    plt.plot(timepoints, np.array(rH_l2optim.K_hat(timepoints, c_l2optim, gamma_l2optim)), '--', label='l2optim kernel')
    plt.legend()
    plt.savefig(r'Output/kernelplot_initial.png')
    plt.close()
    # plt.show()

    # Experiment on l2optim using different amounts of exponential terms
    rH_l2optim.M = 100000
    ns = np.arange(start=1, stop=11)
    deltas = [10, 20, 50, 100, 500]
    # gamma_0 = [1,2,3,5,10,15,20,40,60,100]
    gamma_0 = {1: np.array([1]),
               2: np.array([1, 100]),
               3: np.array([1, 10, 100]),
               4: np.array([1, 3, 10, 100]),
               5: np.array([1, 3, 10, 40, 100]),
               6: np.array([1, 3, 10, 40, 60, 100]),
               7: np.array([1, 3, 10, 20, 40, 60, 100]),
               8: np.array([1, 2, 3, 10, 20, 40, 60, 100]),
               9: np.array([1, 2, 3, 10, 15, 20, 40, 60, 100]),
               10: np.array([1, 2, 3, 5, 10, 15, 20, 40, 60, 100])}
    df_running_times = pd.DataFrame(columns=deltas, index=ns)
    df_running_times_parameters = pd.DataFrame(columns=deltas, index=ns)
    df_running_times_K = pd.DataFrame(columns=deltas, index=ns)

    df_K_hat = pd.DataFrame(index=timepoints)
    for n in ns:
        for delta in deltas[-1:]:
            dt1 = datetime.datetime.now()
            rH_l2optim = rHeston(S0, V0, Lambda, theta, nu, rho, T, Tn, n, M, H)
            c_l2optim, gamma_l2optim = rH_l2optim.optim_l2(delta=1 / delta, n=500, gamma_0=gamma_0[n])
            df_K_hat[str(n) + '_' + str(delta)] = rH_l2optim.K_hat(timepoints, c_l2optim, gamma_l2optim)
            dt2 = datetime.datetime.now()
            df_running_times[delta][n] = (dt2 - dt1).total_seconds()


    plt.scatter(df_running_times.reset_index()['index'], df_running_times[10], label='delta=1/10')
    plt.scatter(df_running_times.reset_index()['index'], df_running_times[20], label='delta=1/20')
    plt.scatter(df_running_times.reset_index()['index'], df_running_times[50], label='delta=1/50')
    plt.scatter(df_running_times.reset_index()['index'], df_running_times[100], label='delta=1/100')
    plt.legend()
    plt.show()

    plt.plot(df_running_times)
    plt.xlabel('Number of exponential terms')
    plt.ylabel('Seconds')
    plt.title('Running times for l2optim')
    plt.show()

    # for n in ns:
    #     plt.plot(df_K_hat[n], label=n)
    # plt.legend()
    # plt.show()

    actual_kernel = rH_l2optim.K(timepoints)

    fig, ax = plt.subplots()
    ax.plot(timepoints, actual_kernel, label='actual kernel', color='dimgray')
    for n in ns:
        ax.plot(timepoints, df_K_hat.filter(regex=str(n)+'_100'), '--', label='n='+str(n))
    ax.legend()
    plt.savefig(r'Output/kernel_l2optim.png')
    plt.close()

    df_errors = (df_K_hat.subtract(actual_kernel, axis='rows'))**2
    sum_errors = df_errors.sum(axis=0)
    df_sum_errors = sum_errors.reset_index()
    df_sum_errors['n'] = df_sum_errors.apply(lambda x: x['index'].split('_')[0], axis=1).astype(int)
    df_sum_errors['delta'] = df_sum_errors.apply(lambda x: x['index'].split('_')[1], axis=1).astype(int)
    df_sum_errors.drop('index', axis=1, inplace=True)

    pivot_errors = pd.pivot_table(df_sum_errors, index=['n'], columns=['delta'], aggfunc='mean').sort_index()

    fig, ax = plt.subplots()
    # ax.scatter(pivot_errors.reset_index()['n'], pivot_errors[(0, 10)], label=r'$\delta$=10')
    ax.scatter(pivot_errors.reset_index()['n'], pivot_errors[(0, 20)], label=r'$\delta$=20')
    ax.scatter(pivot_errors.reset_index()['n'], pivot_errors[(0, 50)], label=r'$\delta$=50')
    ax.scatter(pivot_errors.reset_index()['n'], pivot_errors[(0, 100)], label=r'$\delta$=100')

    ax.legend()
    plt.savefig(r'Output/l2optim_errors.png')

    plt.plot(pivot_errors)

    df_running_times['error'] = sum_errors
    df_running_times.reset_index(inplace=True)
    df_running_times.rename(columns={'index':'n'}, inplace=True)


    # Experiment on Lifting heston varying r_N and N
    r_n = np.arange(start=1.5, stop=3, step = 0.1)
    ns = [10, 20, 50, 100, 200, 500]

    df_running_times = pd.DataFrame(columns=r_n, index=ns)
    df_running_times_parameters = pd.DataFrame(columns=r_n, index=ns)

    df_K_hat = pd.DataFrame(index=timepoints)
    for n in ns:
        for r in r_n:
            dt1 = datetime.datetime.now()
            rH_lifting = rHeston(S0, V0, Lambda, theta, nu, rho, T, Tn, n, M, H)
            c_lifting, gamma_lifting = rH_lifting.set_c_and_gamma_lifting_heston(r)
            df_K_hat[str(n) + '_' + str(r)] = rH_lifting.K_hat(timepoints, c_lifting, gamma_lifting)
            # dt1 = datetime.datetime.now()
            # S, V = rH_lifting.MC_simulate(np.array(c_lifting), np.array(gamma_lifting))
            dt2 = datetime.datetime.now()
            df_running_times[r][n] = (dt2 - dt1).seconds + (dt2 - dt1).microseconds * 1e-6

    df_running_times.to_csv(r'Output/Heston_lifting_running_times.csv')

    fig, ax = plt.subplots()
    ax.scatter(df_running_times.transpose().reset_index()['index'], df_running_times.transpose()[10], label='n=10')
    ax.scatter(df_running_times.transpose().reset_index()['index'], df_running_times.transpose()[20], label='n=20')
    ax.scatter(df_running_times.transpose().reset_index()['index'], df_running_times.transpose()[50], label='n=50')
    ax.scatter(df_running_times.transpose().reset_index()['index'], df_running_times.transpose()[100], label='n=100')
    ax.scatter(df_running_times.transpose().reset_index()['index'], df_running_times.transpose()[200], label='n=200')
    ax.scatter(df_running_times.transpose().reset_index()['index'], df_running_times.transpose()[500], label='n=500')
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0)
    plt.savefig(r'Output/Lifting_running_times.png')
    plt.close()
    # plt.show()

    # Actual kernel
    actual_kernel = rH_lifting.K(timepoints)

    df_errors = (df_K_hat.subtract(actual_kernel, axis='rows'))**2
    sum_errors = df_errors.sum(axis=0)
    df_sum_errors = sum_errors.reset_index()
    df_sum_errors['n'] = df_sum_errors.apply(lambda x: x['index'].split('_')[0], axis=1).astype(int)
    df_sum_errors['r'] = df_sum_errors.apply(lambda x: x['index'].split('_')[1], axis=1)
    df_sum_errors.drop('index', axis=1, inplace=True)

    pivot_errors = pd.pivot_table(df_sum_errors, columns=['n'], index=['r'], aggfunc='sum').sort_index()
    r = [round(float(x),2) for x in pivot_errors.index]

    fig, ax = plt.subplots()
    ax.scatter(r, pivot_errors[(0, 20)], label='n=20')
    ax.scatter(r, pivot_errors[(0, 50)], label='n=50')
    ax.scatter(r, pivot_errors[(0, 100)], label='n=100')
    ax.scatter(r, pivot_errors[(0, 200)], label='n=200')
    ax.scatter(r, pivot_errors[(0, 500)], label='n=500')
    ax.legend()
    plt.savefig(r'Output/errors_Lifting_100.png')
    plt.close()


    # Simulation
    Lambda = 0
    N = 5
    rH_l2optim = rHeston(S0, V0, Lambda, theta, nu, rho, T, Tn, N, M, H)
    c_l2optim, gamma_l2optim = rH_l2optim.optim_l2(delta=1/100, n=500, gamma_0=[1,2,3,10,100])

    S, V = rH_l2optim.MC_simulate(c=c_l2optim, gamma=gamma_l2optim)

    timepoints = rH_l2optim.dt * np.arange(0, rH_l2optim.Tn)
    plt.plot(timepoints, S.mean(axis=0))
    plt.plot(timepoints, S.mean(axis=0) + 1.96 * S.std(axis=0) / np.sqrt(rH_l2optim.M))
    plt.plot(timepoints, S.mean(axis=0) - 1.96 * S.std(axis=0) / np.sqrt(rH_l2optim.M))
    plt.show()


    timepoints = rH_l2optim.dt * np.arange(0, rH_l2optim.Tn)
    plt.plot(timepoints, V.mean(axis=0))
    plt.plot(timepoints, V.mean(axis=0) + 1.96 * V.std(axis=0) / np.sqrt(rH_l2optim.M))
    plt.plot(timepoints, V.mean(axis=0) - 1.96 * V.std(axis=0) / np.sqrt(rH_l2optim.M))
    plt.axhline(V0,linestyle='--')
    plt.show()


    # Simulate underlying price and volatility path
    Lambda = 0.3
    rho = -0.7
    nu = 0.3
    N = 5
    Tn = 1000

    rH = rHeston(S0, V0, Lambda, theta, nu, rho, T, Tn, N, M, H)

    delta = 1/500
    n = 1000
    gamma_0 = np.array([1,3,10,40,100])
    c, gamma = rH.optim_l2(delta,n,gamma_0)

    S, V = rH.MC_simulate(c, gamma)

    # Check the mean of S:
    timepoints = rH_l2optim.dt * np.arange(0, rH_l2optim.Tn)
    plt.plot(timepoints, S.mean(axis=0))
    plt.plot(timepoints, S.mean(axis=0) + 1.96 * S.std(axis=0) / np.sqrt(rH.M))
    plt.plot(timepoints, S.mean(axis=0) - 1.96 * s.std(axis=0) / np.sqrt(rH.M))
    plt.show()

    # Check the mean of V (should be theta):
    plt.plot(timepoints, V.mean(axis=0))
    plt.plot(timepoints, V.mean(axis=0) + 1.96 * V.std(axis=0) / np.sqrt(rH.M))
    plt.plot(timepoints, V.mean(axis=0) - 1.96 * V.std(axis=0) / np.sqrt(rH.M))
    plt.show()

