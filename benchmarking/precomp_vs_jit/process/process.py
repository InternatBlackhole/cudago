#!/usr/bin/env -S python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import os.path as path
import numpy as np
import colorsys
import math

pathToBench = "../results/"

def load_csv(file):
    return pd.read_csv(file, delimiter=';', header='infer')

def load_from_csv(dir, selector):
    df = pd.DataFrame()
    for file in selector:
        dfi = load_csv(path.join(dir, file))
        df = pd.concat([df, dfi]) #ignore_index=True
    return df.reset_index(drop=True)

def save_df_to_csv(df: pd.DataFrame, file):
    df.to_csv(path.join('.', file), sep=';', index=False, header=True)

precomp_dir = path.join(pathToBench, 'precomp')
noprecomp_dir = path.join(pathToBench, 'noprecomp')

df_precomp = load_from_csv(precomp_dir,filter(lambda x: x.endswith('.csv'), os.listdir(precomp_dir)))
df_noprecomp = load_from_csv(noprecomp_dir, filter(lambda x: x.endswith('.csv'), os.listdir(noprecomp_dir)))

df_precomp_means = df_precomp.groupby(['Operation', 'ProblemSize']).mean(numeric_only=True)
df_noprecomp_means = df_noprecomp.groupby(['Operation', 'ProblemSize']).mean(numeric_only=True)

df_precomp_mean_kernel_dur = df_precomp_means.loc['Dur']
df_noprecomp_mean_kernel_dur = df_noprecomp_means.loc['Dur']

df_precomp_mean_call_dur = df_precomp_means.loc['CallDur']
df_noprecomp_mean_call_dur = df_noprecomp_means.loc['CallDur']

df_mean_kernel_durs = pd.merge(
    df_precomp_mean_kernel_dur,
    df_noprecomp_mean_kernel_dur,
    on='ProblemSize',
    suffixes=('_precomp', '_noprecomp')
)

df_mean_call_durs = pd.merge(
    df_precomp_mean_call_dur,
    df_noprecomp_mean_call_dur,
    on='ProblemSize',
    suffixes=('_precomp', '_noprecomp')
)


column_rename = {
    'Time_precomp': 'Statično prevajanje',
    'Time_noprecomp': 'Prevajanje med izvajanjem'
}
index_rename = lambda x: f'$2^{{{int(math.log2(x))}}}$'

figure, [ax1, ax2] = plt.subplots(figsize=(9,6), nrows=2, ncols=1) 

df1 = df_mean_kernel_durs.rename(columns=column_rename, index=index_rename)
df1.plot.bar(ax=ax1, ylabel='Čas (ms)', xlabel='Velikost problema (dolžina polja)', title='Povprečen čas izvajanja ščepca', logy=True, rot=0)

df2 = df_mean_call_durs.rename(columns=column_rename, index=index_rename) / 1000
df2.plot.bar(ax=ax2, ylabel='Čas (ms)', xlabel='Velikost problema (dolžina polja)', title='Povprečen čas klica ščepca', logy=True, rot=0) #\u03BC

major_formatter = plt.FuncFormatter(lambda x, _: f'{x:.3f}'.rstrip('0') if int(x) != x else f'{int(x)}')

ax1.yaxis.set_major_formatter(major_formatter)
ax2.yaxis.set_major_formatter(major_formatter)

figure.tight_layout()
figure.show()
