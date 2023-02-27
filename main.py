import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import func

# MAKE DATAFRAMES

temps_df = pd.read_csv('temp.csv')
temps_df.set_index(temps_df['Date'], inplace=True)
temps_df = temps_df.drop('Date', axis=1)
prcp_df = pd.read_csv('allmonths_pcp.csv')
prcp_df.set_index(prcp_df["Date"], inplace=True)
prcp_df = prcp_df.drop('Date', axis=1)

temps_df[temps_df == -99] = np.NaN
temps_df.dropna(axis = 0, inplace=True)

prcp_df[prcp_df == -99] = np.NaN
prcp_df.dropna(axis=0, inplace=True)

cols = [col[:3] for col in temps_df.columns]
temps_df.columns = cols
prcp_df.columns = cols

# print(temps_df.info())
# print()
# print(prcp_df.info())


def median(data):
    data.sort()
    if len(data) % 2 == 0:
        m = (data[len(data) // 2] + data[len(data) // 2 - 1]) / 2
    else:
        m = data[len(data) // 2]
    return m


def mean(data):
    total = sum(data)
    m = total / len(data)
    return m


def variance(data):
    new_list = [(val - mean(data)) ** 2 for val in data]
    v = mean(new_list)
    return v


def stand_dev(data):
    v = variance(data)
    s = math.sqrt(v)
    return s


def plot_data_bar(df, data):
    if type(data) is str:
        x_data = df.iloc[:, 0]
        y_data = df[data]
        plt.title(f'{data}')
    else:
        x_data = list(df.columns.values)[1:]
        y_data = df.loc[data][1:]
        plt.title(f'{data}')

    plt.bar(x_data, y_data)
    plt.xticks(rotation=90)
    plt.show()


def moving_avg(data, w=None):
    if w is None:
        w = [1, 1, 1, 1]
    new_list = []
    for i in range(len(data) - len(w)+1):
        k = []
        for j in range(len(w)):
            k.append(data[i + j]*w[j])
        new_list.append(sum(k)/sum(w))
    return new_list


def fading_moving_avg(data, w_list):
    output = []
    w1, w2 = w_list[0], w_list[1]
    if sum(w_list) == 1:
        for i in range(len(data)):
            if i == 0:
                output.append(data[i])
            else:
                output.append((w1 * output[i - 1]) + (w2 * data[i]))

    return output


# plot_data_bar(temps_df, 1895)

temps_df['Mean'] = round(temps_df.loc[:,'Jan':'Dec'].mean(axis=1),1)
prcp_df['Mean'] = round(prcp_df.loc[:,'Jan':'Dec'].mean(axis=1),1)

temps_df['Median'] = temps_df.loc[:,'Jan':'Dec'].median(axis=1)
prcp_df['Median'] = prcp_df.loc[:,'Jan':'Dec'].median(axis=1)

temps_df['Std'] = temps_df.loc[:,'Jan':'Dec'].std(axis=1)
prcp_df['Std'] = prcp_df.loc[:,'Jan':'Dec'].std(axis=1)

temps_df['Max'] = temps_df.loc[:, "Jan":'Dec'].max(axis=1)
prcp_df['Max'] = prcp_df.loc[:, "Jan":'Dec'].max(axis=1)

temps_df['Min'] = temps_df.loc[:, "Jan":'Dec'].min(axis=1)
prcp_df['Min'] = prcp_df.loc[:, "Jan":'Dec'].min(axis=1)

mt_df = pd.DataFrame()
mt_df['mean'] = temps_df.loc[:, 'Jan':'Dec'].mean()
mt_df['median'] = temps_df.loc[:, 'Jan':'Dec'].median()
mt_df['std'] = temps_df.loc[:, 'Jan':'Dec'].std()
mt_df['max'] = temps_df.loc[:, 'Jan':'Dec'].max()
mt_df['max dates'] = temps_df.loc[:, 'Jan':'Dec'].idxmax()
mt_df['min'] = temps_df.loc[:, 'Jan':'Dec'].min()
mt_df['min dates'] = temps_df.loc[:, 'Jan':'Dec'].idxmin()

mp_df = pd.DataFrame()
mp_df['mean'] = prcp_df.loc[:, 'Jan':'Dec'].mean()
mp_df['median'] = prcp_df.loc[:, 'Jan':'Dec'].median()
mp_df['std'] = prcp_df.loc[:, 'Jan':'Dec'].std()
mp_df['max'] = prcp_df.loc[:, 'Jan':'Dec'].max()
mp_df['max dates'] = prcp_df.loc[:, 'Jan':'Dec'].idxmax()
mp_df['min'] = prcp_df.loc[:, 'Jan':'Dec'].min()
mp_df['min dates'] = prcp_df.loc[:, 'Jan':'Dec'].idxmin()

# print(mp_df)

# min_temps = temps_df.loc[:, 'Jan':'Dec'].min()
# min_temps_in = temps_df.loc[:, 'Jan':'Dec'].idxmin()
# print(prcp_df.iloc[:, :])


def plot_data(df, data):
    if type(data) is str:
        x_data = df.index.values
        y_data = df[data]
        plt.title(f'{data}')
    else:
        x_data = list(df.columns.values)[1:]
        y_data = df.loc[data][1:]
        plt.title(f'{data}')

    plt.plot(x_data, y_data, '-r')
    plt.xticks(rotation=90)
    plt.show()

#
# plot_data(temps_df, 'Mean')
# plot_data(temps_df, 'Median')
# plot_data(temps_df, 'Std')
# plot_data(temps_df, 'Max')
# plot_data(prcp_df, 'Min')
# plot_data(prcp_df, 'Mean')
# plot_data(prcp_df, 'Median')
# plot_data(prcp_df, 'Std')
# plot_data(prcp_df, 'Max')
# plot_data(prcp_df, 'Min')


def plot_data_ma(df, data):
    if type(data) is str:
        x_data = df.index.values[9:-10]
        y_data = df[data].values
        y_data = moving_avg(y_data, [1 for i in range(20)])
        plt.title(f'{data}')
    else:
        x_data = list(df.columns.values)[1:]
        y_data = df.loc[data][1:]
        plt.title(f'{data}')

    plt.plot(x_data, y_data, '-r')
    plt.plot(df.index.values, df[data].values)
    plt.xticks(rotation=90)
    plt.show()


def plot_data_bar_ed(x_data, y_data2, y_data, title):
    plt.title(title)
    plt.ylim(min(y_data)-30, max(y_data)+30)
    width = .25
    years = [x for x in range(1, 13)]
    years = np.array(years)
    plt.bar(years+width, y_data, width=width)
    plt.bar(years-width, y_data2, width=width)
    # plt.xticks(ticks=x_data)
    plt.xticks(rotation=90)
    plt.show()


plot_data_ma(temps_df, 'Mean')
# plot_data_ma(temps_df, 'Min')
# plot_data_ma(temps_df, 'Max')
# plot_data_ma(temps_df, 'Median')
#
plot_data_ma(prcp_df, 'Mean')
# plot_data_ma(prcp_df, 'Min')
# plot_data_ma(prcp_df, 'Max')
# plot_data_ma(prcp_df, 'Median')

plot_data_bar_ed(mp_df.index.values, mp_df['max dates'].values, mp_df['min dates'].values, 'Min to Max Precipitation')
plot_data_bar_ed(mt_df.index.values, mt_df['max dates'].values, mt_df['min dates'].values, 'Min to Max Temperature')
temps_ma = moving_avg(temps_df.loc[:, 'Mean'].values.tolist(), [1 for i in range(20)])
prcp_ma = moving_avg(prcp_df.loc[:, 'Mean'].values.tolist(), [1 for k in range(20)])

years_l1 = [i+.5 for i in range(1904, 1943)]
temps_l1 = temps_ma[:len(years_l1)]

years_l2 = [i+.5 for i in range(1943, 1980)]
temps_l2 = temps_ma[len(years_l1):len(years_l1)+len(years_l2)]
# print(len(years_l2), len(temps_l2))

years_l3 = [i+.5 for i in range(1980, 2013)]
temps_l3 = temps_ma[len(years_l1)+len(years_l2)-1:]

prcp_l3 = prcp_ma[len(years_l1)+len(years_l2)-1:]
# print(len(years_l3), len(temps_l3))

#
# temps_l2 = temps_df.loc[1943:1979, 'Mean'].values.tolist()
# years_l2 = temps_df.loc[1943:1979].index.tolist()
#
# temps_l3 = temps_df.loc[1980:, 'Mean'].values.tolist()
# years_l3 = temps_df.loc[1980:2002].index.tolist()

lst1 = func.least_sqrs(years_l3, temps_l3)
# print(lst1)
x1 = 2025
x2 = 2030
x3 = 2035
es_2025 = lst1[0]*x1+lst1[1]
es_2030 = lst1[0]*x2+lst1[1]
es_2035 = lst1[0]*x3+lst1[1]
print(es_2025, es_2030, es_2035)
cor_l1 = func.corcoeff(years_l1, temps_l1)
cor_l2 = func.corcoeff(years_l2, temps_l2)
cor_l3 = func.corcoeff(years_l3, temps_l3)
print(cor_l1, cor_l2, cor_l3)
cor_p_t = func.corcoeff(temps_l3, prcp_l3)
print(cor_p_t)

