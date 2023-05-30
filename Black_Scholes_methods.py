# Imports
import pandas as pd
import numpy as np
import scipy
import math

def BlackScholesCallPut(S, K, T, sigma, r, q, call_put=1):
    d1 = (np.log(S/K) + (r+.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return call_put*(S*scipy.stats.norm.cdf(call_put*d1)*np.exp(-q*T) - K*np.exp (-r*T) * scipy.stats.norm.cdf (call_put*d2))

def sum_parity(x0, df_call_put, TTM):
    q = x0[0]
    TTM1 = [t[0] for t in TTM]
    TTM2 = [t[1] for t in TTM]
    r = pd.DataFrame({'quote_date': TTM1, 'TTM': TTM2,'r': x0[1:]})
    sum = df_call_put.apply(lambda x: (x['diff']-(np.exp(-q*x['TTM'])*x['mid_underlying']-np.exp(-r[(r['TTM'] == x['TTM'])&(r['quote_date'] == x['quote_date'])]['r'].values*x['TTM'])*x['strike']))**2,axis=1).sum()
    print(sum)
    return sum

def get_q_and_r(df_call_put):
    # List of unique sets of quote_date and TTM
    TTM = list(set([(df_call_put.quote_date.iloc[i], df_call_put.TTM.iloc[i]) for i in range(len(df_call_put))]))
    # define initial guess
    q_init = 0.01977683054190561
    r_init = np.arange(-0.01, 0.02, step=0.03/len(TTM))
    # r_init = [0.02] * len(TTM)
    x0 = np.append([q_init], r_init)

    # define bounds for q and each element of r_init
    q_bounds = (0, 0.1)
    r_bounds = [(-0.1, 0.1) for i in range(len(r_init))]

    # concatenate bounds into a single tuple
    bounds = (q_bounds,) + tuple(r_bounds)
    solution = scipy.optimize.minimize(sum_parity, x0, bounds=bounds, args=(df_call_put, TTM), method='L-BFGS-B')

    q = solution['x'][0]
    r = solution['x'][1:]

    return q, r

def combine_put_call(df):
    df_call = pd.DataFrame(df[df.option_type == 'C'])#.set_index(['quote_date','TTM', 'strike', 'mid_underlying'])['mid_price'], columns=['call_price'])
    df_put = pd.DataFrame(df[df.option_type == 'P'])#.set_index(['quote_date','TTM', 'strike', 'mid_underlying'])['mid_price'], columns=['put_price'])

    df_put_call = df_call.merge(df_put, on=['quote_date', 'TTM', 'strike', 'mid_underlying'])
    df_put_call.rename({'mid_price_x': 'call_price', 'mid_price_y': 'put_price'}, inplace=True, axis='columns')

    df_put_call.drop(['option_type_x','option_type_y'], axis=1, inplace=True)

    return df_put_call

def black_scholes_call(S, K, T, r, q, sigma):
    """Calculates the Black-Scholes call option price"""
    d1 = (math.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N1 = scipy.stats.norm.cdf(d1)
    N2 = scipy.stats.norm.cdf(d2)
    call_price = S * math.exp(-q * T) * N1 - K * math.exp(-r * T) * N2
    return call_price


def implied_volatility(S, K, T, r, q, price):
    """Calculates the implied volatility given call price and other parameters"""
    tol = 1e-6
    max_iter = 200
    sigma = 0.5  # initial guess
    for i in range(max_iter):
        price_diff = price - black_scholes_call(S, K, T, r, q, sigma)
        if abs(price_diff) < tol:
            return sigma
        vega = S * math.exp(-q * T) * scipy.stats.norm.pdf((math.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))) * math.sqrt(T)
        sigma += price_diff / vega
    return np.nan
    # raise ValueError("Implied volatility not found after {} iterations".format(max_iter))


