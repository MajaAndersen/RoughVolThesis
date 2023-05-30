# Imports
import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd

from Black_Scholes_methods import *

# Load data
df_data = pd.read_csv(r'data/UnderlyingOptionsEODQuotes_2019-05-14.csv')
df_data = df_data[df_data.root == 'SPX']

df_data['expiration'] = pd.to_datetime(df_data['expiration'])
df_data['quote_date'] = pd.to_datetime(df_data['quote_date'])

df_data['TTM'] = df_data.apply(lambda x: (x['expiration'] - x['quote_date']).days / 365, axis=1)
df_data['mid_price'] = df_data.apply(lambda x: (x['ask_1545'] + x['bid_1545'])/2, axis=1)
df_data['mid_underlying'] = df_data.apply(lambda x: (x['underlying_ask_1545'] + x['underlying_bid_1545'])/2, axis=1)

df_data = df_data[df_data.bid_1545 > 0]

df_spx = df_data[['quote_date','TTM', 'mid_price', 'mid_underlying', 'strike', 'option_type']]
df_spx['option_type'] = df_spx.option_type.str.upper()
df_call = df_spx[df_spx.option_type == 'C']
df_call_put = combine_put_call(df_spx)
df_call_put['diff'] = df_call_put.apply(lambda x: x['call_price']-x['put_price'], axis=1)

del df_data
del df_spx

# Use put call parity to get q and r
q, r = get_q_and_r(df_call_put)
TTM = df_call_put.TTM.unique()
df_q_r_TTM = pd.DataFrame({'TTM': TTM, 'q': [q]*len(TTM), 'r': r})

df_test = df_call.merge(df_q_r_TTM, on='TTM', how='left')

df_test['dist'] = df_test.apply(lambda x: (x['strike'] - x['mid_underlying']), axis=1)
df_test['imp_vol'] = df_test.apply(lambda x: implied_volatility(x['mid_underlying'], x['strike'], x['TTM'], x['r'], x['q'], x['mid_price']), axis=1)

df_test['log_moneyness'] = df_test.apply(lambda x: np.log(x['strike']/x['mid_underlying']), axis=1)

# Implied volatility
pivot_impvol = pd.pivot_table(df_test, values='imp_vol', columns='TTM', index='log_moneyness')

T_label = [round(t, 3) for t in pivot_impvol.columns]
colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(pivot_impvol.iloc[0])))
for i in range(len(pivot_impvol.columns)):
    t = pivot_impvol.columns[i]
    plt.scatter(pivot_impvol[t].index, pivot_impvol[t], label=t, s=[8*1**n for n in range(len(pivot_impvol.iloc[:,i]))], color=colors[i])
plt.legend(T_label)
plt.xlabel('log(moneyness)')
plt.ylabel(r'$\sigma_{BS}$')
plt.show()

# ATM skew
df_test2 = df_test[df_test.imp_vol.isna()==False]

TTM = df_test2.TTM.unique()
ATM = np.zeros(len(TTM))
for i in range(len(TTM)):
    t = TTM[i]
    df_test3 = df_test2[df_test2.TTM == t].sort_values('k')
    iv_l = df_test3[df_test3.k < 0].iloc[-1]
    iv_h = df_test3[df_test3.k > 0].iloc[0]
    ATM[i] = (iv_h.imp_vol - iv_l.imp_vol)/(iv_h.k - iv_l.k)

b, a = scipy.stats.linregress(np.log(TTM), np.log(abs(ATM)))[:2]
print('t^' + str(b))
np.exp(a)
power_law_fit = lambda x: np.exp(a)*x**(b)

x = np.arange(start=TTM[0], stop=TTM[-1], step=0.01)
text = r'${}\cdot t^{{ {}{:.3f} }}$'.format(round(np.exp(a),3), '+' if np.sign(b) > 0 else '-', abs(b))
plt.plot(x, power_law_fit(x), label='Power law fit: ' + text, color='cornflowerblue')
plt.scatter(TTM, abs(ATM), label='Empirical observations', color='red')
plt.legend()
plt.show()