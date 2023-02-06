import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# MAKE DATAFRAMES

temps_df = pd.read_csv('temp.csv')
temps_df.set_index(temps_df['Date'], inplace=True)
temps_df = temps_df.drop('Date', axis=1)
prcp_df = pd.read_csv('prcp.csv')
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
mt_df['min'] = temps_df.loc[:, 'Jan':'Dec'].min()

mp_df = pd.DataFrame()
mp_df['mean'] = prcp_df.loc[:, 'Jan':'Dec'].mean()
mp_df['median'] = prcp_df.loc[:, 'Jan':'Dec'].median()
mp_df['std'] = prcp_df.loc[:, 'Jan':'Dec'].std()
mp_df['max'] = prcp_df.loc[:, 'Jan':'Dec'].max()
mp_df['min'] = prcp_df.loc[:, 'Jan':'Dec'].min()

# print(mt_df)
# print(temps_df.iloc[:, :])
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
        x_data = df.index.values[3:]
        y_data = df[data].values
        y_data = moving_avg(y_data)
        plt.title(f'{data}')
    else:
        x_data = list(df.columns.values)[1:]
        y_data = df.loc[data][1:]
        plt.title(f'{data}')

    plt.plot(x_data, y_data, '-r')
    plt.xticks(rotation=90)
    plt.show()


plot_data_ma(temps_df, 'Mean')







