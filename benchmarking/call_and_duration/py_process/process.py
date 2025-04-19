#!/bin/python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import os.path as path
import colorsys

def load_csv(file):
    return pd.read_csv(file, delimiter=';', header='infer', index_col='Test') #index_col='Image'

def load_from_csv_in_dir(dir):
    df = pd.DataFrame()
    for file in filter(lambda x: x.endswith(".csv"), os.listdir(dir)):
        dfi = load_csv(path.join(dir, file))
        df = pd.concat([df, dfi])
    return df


def generate_unique_colors(n):
    """
    Generates n unique colors in RGB format, evenly spaced rainbow gradient.
    """
    HSV_tuples = [(x*1.0/n, 0.9, 0.9) for x in range(n)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    return RGB_tuples

bench_dir = "../benches"

df_go = load_from_csv_in_dir( path.join(bench_dir, 'go_results'))
df_native = load_from_csv_in_dir(path.join(bench_dir, 'native_results'))

df_go_op_test_grouping = df_go.groupby(['Operation', 'Test'])
df_native_op_test_grouping = df_native.groupby(['Operation', 'Test'])

df_go_mean_time_per_op_test = df_go_op_test_grouping.mean(numeric_only=True)
df_native_mean_time_per_op_test = df_native_op_test_grouping.mean(numeric_only=True)

df_means = pd.merge(
    df_go_mean_time_per_op_test,
    df_native_mean_time_per_op_test,
    left_index=True,
    right_index=True,
    sort=True,
    how='inner',
    suffixes=('Go', 'Native'),
)
df_means['TimeRatio'] = df_means['TimeGo'] / df_means['TimeNative']

df_means_durs = df_means.loc[df_means.index.get_level_values('Operation').str.match(pat=r'^.*KernelDur$')]
df_means_calls = df_means.loc[df_means.index.get_level_values('Operation').str.match(pat=r'^.*KernelCall$')]

def scatter_plot_native_go_relation(ax: plt.Axes, *dfs: tuple[pd.DataFrame, str], x: str, y: str, x_label: str, y_label: str, title: str):
    for color, (df, label) in zip(generate_unique_colors(len(dfs)), dfs):
        df.plot.scatter(x=x, y=y, ax=ax, s=20, color=color, label=label)
    
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    
    ax.grid(axis='both',c='0.9')
    ax.set_axisbelow(True)

def new_scatter_relation_figure(**kwargs) -> tuple[plt.Figure, plt.Axes]:
    (fig, ax) = plt.subplots(**kwargs)
    return fig, ax

figure, ax = new_scatter_relation_figure(figsize=(9,6))
ax.set_yscale('log')
ax.set_xscale('log')

scatter_plot_native_go_relation(
    ax,
    (df_means_durs.loc['Image_KernelDur'], "Iskanje robov"),
    (df_means_durs.loc['Sort_KernelDur'], "Urejanje"),
    (df_means_durs.loc['AddToAll_KernelDur'], "Prištevanje"),

    #(df_means_calls.loc['Image_KernelCall'], "Iskanje robov"),
    #(df_means_calls.loc[df_means_calls.index.get_level_values('Operation').str.match(pat=r"^Sort_.*KernelCall$")], "Urejanje"),
    ##(df_means_calls.loc['Sort_GoStartKernelCall'], "Urejanje (začetni)"),
    ##(df_means_calls.loc['Sort_GoMiddleKernelCall'], "Urejanje (vmesni)"),
    ##(df_means_calls.loc['Sort_GoFinishKernelCall'], "Urejanje (končni)"),
    #(df_means_calls.loc['AddToAll_KernelCall'], "Prištevanje"),

    #title='Average kernel execution duration',
    #title= 'Average kernel call duration',
    title='Povprečni čas izvajanja ščepca',
    #title='Povprečni čas klica ščepca',
    x_label='CUDA C++ (ms)',
    y_label='Ovojnica go (ms)',
    x='TimeNative',
    y='TimeGo',
)

xbounds = list(ax.get_xlim())
x = np.arange(xbounds[0], xbounds[1], 1)
y = x
ax.plot(x, y, ls='--', color='.3', zorder=-100)
#ax.axis('equal')

figure.tight_layout()
figure.show()
