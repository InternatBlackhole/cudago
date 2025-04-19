#!/usr/bin/env -S python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import os.path as path
import numpy as np
from sklearn.linear_model import LinearRegression

pathToBench = "../bench/"

def load_csv(file):
    return pd.read_csv(file, delimiter=';', header='infer') #index_col='Image'

def load_from_csv(dir, selector):
    df = pd.DataFrame()
    for file in selector:
        dfi = load_csv(path.join(dir, file))
        df = pd.concat([df, dfi]) #ignore_index=True
    return df.reset_index(drop=True)

def save_df_to_csv(df: pd.DataFrame, file):
    df.to_csv(path.join('.', file), sep=';', index=False, header=True)

go_dir = path.join(pathToBench, 'go')
native_dir = path.join(pathToBench, 'native')

df_go = load_from_csv(go_dir, filter(lambda x: x.endswith('.csv'), os.listdir(go_dir)))
df_native = load_from_csv(native_dir, filter(lambda x: x.endswith('.csv'), os.listdir(native_dir)))

for (first, second) in zip(df_go['Func'].unique(), df_native['Func'].unique()):
    if first != second:
        print('Go and CUDA C functions are not the same')
        sys.exit(1)

df_go_grouped = df_go.groupby(['Func'])
df_go_mean_by_runs = df_go_grouped.mean().loc[df_go['Func'].unique()].reset_index()
df_go_mean_by_runs['Func'] = df_go_mean_by_runs['Func'].apply(lambda x: int(x[3:]))

df_native_grouped = df_native.groupby(['Func'])
df_native_mean_by_runs = df_native_grouped.mean().loc[df_native['Func'].unique()].reset_index()
df_native_mean_by_runs['Func'] = df_native_mean_by_runs['Func'].apply(lambda x: int(x[3:]))

go_model = LinearRegression()
native_model = LinearRegression()

go_model.fit(df_go_mean_by_runs['Func'].values.reshape(-1, 1), df_go_mean_by_runs['Time'].values.reshape(-1, 1))
native_model.fit(df_native_mean_by_runs['Func'].values.reshape(-1, 1), df_native_mean_by_runs['Time'].values.reshape(-1, 1))

figure, ax = plt.subplots(figsize=(9,6))

df_native_mean_by_runs.plot.scatter(x='Func', y='Time', c='red', label='CUDA C++', ax=ax)
df_go_mean_by_runs.plot.scatter(x='Func', y='Time', c='blue', label='Ovojnica go', ax=ax)


ax.plot(df_go_mean_by_runs['Func'], go_model.predict(df_go_mean_by_runs['Func'].values.reshape(-1, 1)), c='crimson', label='Regresija ovojnice Go')
ax.plot(df_native_mean_by_runs['Func'], native_model.predict(df_native_mean_by_runs['Func'].values.reshape(-1, 1)), c='green', label='Regresija CUDA C++')

ax.set_ylabel('Povprečen čas (\u03BCs)')
ax.set_xlabel('Število parametrov v ščepcu')
ax.set_title('Povprečen čas klica ščepca')
ax.legend()
ax.grid(axis='both', c='0.9')
ax.set_axisbelow(True)

figure.tight_layout()
figure.show()

#df_go.plot.box(by='Func', column='Time', ax=ax1)
#df_native.plot.box(by='Func', column='Time', ax=ax2)

#df_go_grouped.plot.box(y='Time', ax=ax1)
#df_native_grouped.plot.box(y='Time', ax=ax2)
