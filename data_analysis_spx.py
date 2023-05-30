import datetime

import pandas as pd
import numpy as np
from datetime import datetime as dt
import scipy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Load data
data = pd.read_csv('data/oxfordmanrealizedvolatilityindices.csv')

rv5 = data['rv5']

rv5_log = data['rv5_ss'].apply(lambda x: np.log(x))

# SP500 data
df_spx = data[data.Symbol == '.SPX']
df_spx['Date'] = [pd.to_datetime(x) for x in df_spx['Unnamed: 0'].apply(lambda x: x[:10])]
df_spx['rv5_log'] = df_spx.apply(lambda x: np.log(np.sqrt(x['rv5'])), axis=1)


# Plot volatilities
fig, ax = plt.subplots()
ax.plot(df_spx['Date'], df_spx['rv5_ss'])
ax.xaxis.set_major_locator(mdates.YearLocator(1))
plt.show()

### Next we want to plot the moments
# Function setup for delta days differences
def fnc(data:pd.Series,x:list, q:float):
    return [np.mean(np.abs(data - data.shift(delta)) ** q) for delta in x]

# Defining the moments
q = [0.5, 1, 1.5, 2, 3]

x = np.arange(1,50)

# Create dataframe to save values of m in
df_m = pd.DataFrame(data=np.log(x), columns=['logx'])
# Calculate the values for each q
for i in q:
    df_m[i] = np.log(fnc(df_spx['rv5_log'], x, i))


fig, ax = plt.subplots(1, 1, figsize = (16,8), dpi = 100)
colours = {0.5:'red', 1:'yellow', 1.5:'green', 2:'blue', 3:'purple'}
for j in range(len(q)):
    i = q[j]
    # Perform linear regression
    coefficients = np.polyfit(df_m['logx'], df_m[i], 1)
    m = coefficients[0]
    b = coefficients[1]
    regression_line = m * df_m['logx'] + b
    ax.plot(df_m['logx'], df_m[i], 'o', color=colours[i])
    ax.plot(df_m['logx'], regression_line, '-', color=colours[i])
    ax.lines[2*j].set_label('q='+str(i))

ax.legend(fontsize='xx-large')

# Set the font size of the x- and y-axes tick labels
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('log h', fontsize=16)
ax.set_ylabel('log m(q,h)', fontsize=16)
plt.show()
# plt.savefig('m-plot.png')

### Next we want to estimate the Hurst parameter from the scaling function
model = []
for i in q:
    z = LinearRegression().fit(np.log(x).reshape(-1,1), np.log(fnc(df_spx['rv5_log'], x, i)))
    model.append(z.coef_)
model = pd.DataFrame(np.transpose(model))

fig, ax = plt.subplots(layout='tight')
b, a = np.polyfit(q, model.loc[0,:], 1)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.5, 0.43, r"$\zeta_q = {} $* q".format(round(b,4)), fontsize=16, bbox=props)
plt.plot(q, b*np.array(q)+a, label='OLS fit', linewidth=2.5)
plt.scatter(q, model.transpose()[0], label='Actual', linewidth=2.3)
plt.ylabel('$\zeta_q$', fontsize=16)
plt.xlabel('$q$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc='lower right', fontsize=16)
plt.show()
# plt.savefig('Hplot.png')


# Do the same but looking at one year at a time
# Create list of dataframes for each year
years = list(set([x.year for x in df_spx.Date]))
df_dict = dict.fromkeys(years)

df_H = pd.DataFrame(columns=['year', 'H', 'R'])

colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(years)))
fig, ax = plt.subplots(figsize=(12,12))

y_len = 252

def H_func(y):
    model = []
    for i in q:
        z = LinearRegression().fit(np.log(x).reshape(-1, 1), pd.Series(np.log(fnc(y, x, i))))
        model.append(z.coef_)
    model = pd.DataFrame(np.transpose(model))
    b, a, r = scipy.stats.linregress(q, model.loc[0, :])[:3]
    return b, a, r

H_func(df_spx['rv5_log'])

# df_spx.reset_index(drop=True, inplace=True)

df_rolled = pd.DataFrame([H_func(df['rv5_log']) for df in df_spx.rolling(y_len) if len(df)==y_len])

df_rolled.columns = ['H', 'intersect', 'R_squared']

plt.plot(df_spx['Date'][251:], df_rolled['H'], label='H')
plt.xlabel('Year')
plt.ylabel('H')
plt.title('H values through time')
plt.savefig(r'Output/H_through_time.png')

plt.plot(df_spx['Date'][251:], df_rolled['intersect'], label='intersect')
plt.xlabel('Year')
plt.ylabel('Intersect')
plt.legend()
plt.savefig(r'Output/intersect_through_time.png')

plt.plot(df_spx['Date'][251:], df_rolled['R_squared'])
plt.xlabel('Year')
plt.ylabel('$R^2$')
plt.savefig(r'Output/R2_through_time.png')


H_rolling = df_spx['rv5_log'].rolling(y_len).apply(H_func)
plt.plot(H_rolling)

plt.plot(df_spx['Date'][251:], H_rolling.reset_index(drop=True)[251:])
plt.xlabel('Year')
plt.ylabel('H')
plt.title('H values through time')
plt.show()

plt.hist(H_rolling)
plt.show()
plt.show()

for year in years:
    model = []

    mask = (df_spx.Date >= dt(year,1,1))&(df_spx.Date < dt(year+1,1,1))
    df_y = df_spx[mask].reset_index()['rv5_log']

    if year == max(years):
        x = x[:29]
    for i in q:
        z = LinearRegression().fit(np.log(x).reshape(-1,1), pd.Series(np.log(fnc(df_y, x, i))))
        model.append(z.coef_)
    model = pd.DataFrame(np.transpose(model))
    b, a, r = scipy.stats.linregress(q, model.loc[0,:])[:3]
    # b, a = np.polyfit(q, model.loc[0,:], 1)

    df_H.loc[year - min(years)] = [year, b, r]

    if year == max(years):
        ax.plot(q, b * np.array(q) + a, c='black')
    else:
        ax.plot(q, b*np.array(q)+a, c=colors[year - min(years)])
    ax.lines[year - min(years)].set_label(year)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel('q', fontsize=14)
ax.set_ylabel('$\zeta_q$', fontsize=14)
plt.savefig('hplot_year_divide.png')

df_H['year'] = df_H['year'].astype('str')
plt.plot(df_H['year'][:-1], df_H['H'][:-1])
plt.show()
#
# df_H
# model = []
# for i in q:
#     z = LinearRegression().fit(np.log(x).reshape(-1,1), np.log(fnc(x, i)))
#     model.append(z.coef_)
# model = pd.DataFrame(np.transpose(model))
#
# fig, ax = plt.subplots(1, 1, figsize = (15,8), dpi = 100)
# b, a = np.polyfit(q, model.loc[0,:], 1)
#
#
# plt.show()
# #Plotting the points with linear regressions on each
# fig, ax = plt.subplots(1, 1, figsize = (16,8), dpi = 100)
# for i in q:
#     ax = sns.regplot(x='x', y='q = {}'.format(i), data=df_plot)
# plt.legend(labels=['q = 0.5','','','q = 1','','','q = 1.5','','','q = 2','','','q = 3','',''])
# plt.xlabel('$\log (\Delta)$')
# plt.ylabel('$\log (m(q,\Delta))$')
# plt.title('spx')
# # plt.savefig('moment_spx.png')
# plt.show()

#
# ### Next we want to do "H-plot"
# model = []
# for i in q:
#     z = LinearRegression().fit(np.log(x).reshape(-1,1), np.log(fnc(x, i)))
#     model.append(z.coef_)
# model = pd.DataFrame(np.transpose(model))
#
# fig, ax = plt.subplots(1, 1, figsize = (15,8), dpi = 100)
# b, a = np.polyfit(q, model.loc[0,:], 1)
# plt.scatter(q, model)
# plt.plot(q, b*np.array(q)+a)
# plt.title('C20 Hurst parameter = {:.3f}'.format(b))
# plt.xlabel('$q$')
# #plt.savefig('hurst_c20.png')
# plt.show()


### Plot histograms
fig, axs = plt.subplots(2,2)
axs[0][0].hist(df_spx['rv5_log'].diff(1)[1:].reset_index(drop=True), bins = 100)
axs[0][0].set_xlim([-2,2])
axs[0][0].set_title('1 day')
axs[0][1].hist(df_spx['rv5_log'].diff(5)[1:].reset_index(drop=True), bins = 100)
axs[0][1].set_xlim([-2,2])
axs[0][1].set_title('5 day')
axs[1][0].hist(df_spx['rv5_log'].diff(25)[1:].reset_index(drop=True), bins = 100)
axs[1][0].set_xlim([-2,2])
axs[1][0].set_title('25 day')
axs[1][1].hist(df_spx['rv5_log'].diff(125)[1:].reset_index(drop=True), bins = 100)
axs[1][1].set_xlim([-2,2])
axs[1][1].set_title('125 day')

plt.show()


acf = sm.tsa.stattools.acf(df_spx['rv5_log'], nlags=200)
lags = sm.add_constant(np.log(range(20,201)))
model = sm.OLS(np.log(acf[20:]), lags)
result = model.fit()
print(result.summary())

fig, ax = plt.subplots()
ax.plot(np.log(range(20,201)), np.log(acf[20:]), marker='o')
ax.plot(np.log(range(20,201)), result.params[1] * np.log(range(20,201)) + result.params[0], color='red')
ax.set_xlabel(r'$\log(h)$')
ax.set_ylabel(r'$\log(\rho(h))$')
plt.savefig(r'Output/data_autocorrolation_widen.png')

def some_fn(df_):
    """
    When iterating over a rolling window it disregards the min_periods
    argument of rolling and will produce DataFrames for all windows

    The input is also of type DataFrame not Series

    You are completely responsible for doing all operations here,
    including ignoring values if the input is not of the correct shape
    or format

    :param df_: A DataFrame produced by rolling
    :return: a column joined, and the max value within the window
    """
    return ','.join(df_['a']), df_['a'].max()


df = pd.DataFrame({'a': list('abdesfkm')})
window = 2
results = pd.DataFrame([some_fn(df_) for df_ in df.rolling(window)])

some_fn(df)