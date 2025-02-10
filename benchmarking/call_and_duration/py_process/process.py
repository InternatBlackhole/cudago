#!/usr/bin/env -S python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import os.path as path
import colorsys

def load_csv(file):
    return pd.read_csv(file, delimiter=';', header='infer') #index_col='Image'

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
df_go_img_op_grouping = df_go.groupby(['Image', 'Operation'])
df_go_mean_time_per_img_op = df_go_img_op_grouping.mean(numeric_only=True).reset_index()
df_go_mean_time_kernel_duration = df_go_mean_time_per_img_op[df_go_mean_time_per_img_op.Operation == 'KernelCall'].drop(columns=['Operation'])
df_go_mean_time_kernel_call = df_go_mean_time_per_img_op[df_go_mean_time_per_img_op.Operation == 'GoStartToEndKernelCall']
df_go_mean_time_kernel_call_mean = df_go_mean_time_kernel_call.groupby('Image').mean(numeric_only=True).reset_index()


df_native = load_from_csv_in_dir(path.join(bench_dir, 'native_results'))
df_native_img_op_grouping = df_native.groupby(['Image', 'Operation'])
df_native_mean_time_per_img_op = df_native_img_op_grouping.mean(numeric_only=True).reset_index()
df_native_mean_time_kernel_duration = df_native_mean_time_per_img_op[df_native_mean_time_per_img_op.Operation == 'KernelCall'].drop(columns=['Operation'])
df_native_mean_time_kernel_call = df_native_mean_time_per_img_op[df_native_mean_time_per_img_op.Operation == 'GoStartToEndKernelCall']
df_native_mean_time_kernel_call_mean = df_native_mean_time_kernel_call.groupby('Image').mean(numeric_only=True).reset_index()

df_go_no_img_stuff = df_go.drop(columns='Image')
df_native_no_img_stuff = df_native.drop(columns='Image')

df_go_sorting_kernel_call = df_go_no_img_stuff[df_go_no_img_stuff.Operation.str.startswith('Sort_Go')]
df_native_sorting_kernel_call = df_native_no_img_stuff[df_native_no_img_stuff.Operation.str.startswith('Sort_Go')]

df_go_sorting_kernel_call_mean = df_go_sorting_kernel_call.groupby('Operation').mean(numeric_only=True)
df_native_sorting_kernel_call_mean = df_native_sorting_kernel_call.groupby('Operation').mean(numeric_only=True)

df_go_addall_kernel_call = df_go_no_img_stuff[df_go_no_img_stuff.Operation.str.startswith('AddToAll_Go')]
df_native_addall_kernel_call = df_native_no_img_stuff[df_native_no_img_stuff.Operation.str.startswith('AddToAll_Go')]

df_go_addall_kernel_call_mean = df_go_addall_kernel_call.groupby('Operation').mean(numeric_only=True)
df_native_addall_kernel_call_mean = df_native_addall_kernel_call.groupby('Operation').mean(numeric_only=True)

df_go_sorting_kernel_duration = df_go_no_img_stuff[df_go_no_img_stuff.Operation == 'Sort_KernelCall']
df_native_sorting_kernel_duration = df_native_no_img_stuff[df_native_no_img_stuff.Operation == 'Sort_KernelCall']

df_go_addall_kernel_duration = df_go_no_img_stuff[df_go_no_img_stuff.Operation == 'AddToAll_KernelCall']
df_native_addall_kernel_duration = df_native_no_img_stuff[df_native_no_img_stuff.Operation == 'AddToAll_KernelCall']

#df_go_sorting_kernel_duration_mean = df_go_sorting_kernel_duration.groupby('Operation').mean(numeric_only=True)
#df_native_sorting_kernel_duration_mean = df_native_sorting_kernel_duration.groupby('Operation').mean(numeric_only=True)

df_kernel_call_means = pd.merge(
    df_go_mean_time_kernel_call_mean,
    df_native_mean_time_kernel_call_mean,
    on='Image',
    how='inner',
    suffixes=('Go', 'Native'),
)
df_kernel_call_means['TimeRatio'] = df_kernel_call_means['TimeGo'] / df_kernel_call_means['TimeNative']

df_kernel_duration_means = pd.merge(
    df_go_mean_time_kernel_duration,
    df_native_mean_time_kernel_duration,
    on='Image',
    how='inner',
    suffixes=('Go', 'Native'),
)
df_kernel_duration_means['TimeRatio'] = df_kernel_duration_means['TimeGo'] / df_kernel_duration_means['TimeNative']

#df_sorting_kernel_call_means = pd.merge(
#    df_go_sorting_kernel_call_mean,
#    df_native_sorting_kernel_call_mean,
#    on='Operation',
#    how='inner',
#    suffixes=('Go', 'Native'),
#)
#df_sorting_kernel_call_means['TimeRatio'] = df_sorting_kernel_call_means['TimeGo'] / df_sorting_kernel_call_means['TimeNative']

df_sorting_kernel_call_means = pd.concat(
    [df_go_sorting_kernel_call.rename(columns={'Time': 'TimeGo'}),
    df_native_sorting_kernel_call.rename(columns={'Time': 'TimeNative'}).drop(columns='Operation')],
    axis=1)
#df_sorting_kernel_call_means = df_sorting_kernel_call_means.groupby('Operation').mean(numeric_only=True)
df_sorting_kernel_call_means['TimeRatio'] = df_sorting_kernel_call_means['TimeGo'] / df_sorting_kernel_call_means['TimeNative']

df_addall_kernel_call_means = pd.concat(
    [df_go_addall_kernel_call.rename(columns={'Time': 'TimeGo'}),
    df_native_addall_kernel_call.rename(columns={'Time': 'TimeNative'}).drop(columns='Operation')],
    axis=1)
df_addall_kernel_call_means['TimeRatio'] = df_addall_kernel_call_means['TimeGo'] / df_addall_kernel_call_means['TimeNative']

#df_addall_kernel_call_means = pd.merge(
#    df_go_addall_kernel_call_mean,
#    df_native_addall_kernel_call_mean,
#    on='Operation',
#    how='inner',
#    suffixes=('Go', 'Native'),
#)
#df_addall_kernel_call_means['TimeRatio'] = df_addall_kernel_call_means['TimeGo'] / df_addall_kernel_call_means['TimeNative']

df_sorting_kernel_duration = pd.concat(
    [df_go_sorting_kernel_duration['Time'].rename('TimeGo'),
    df_native_sorting_kernel_duration['Time'].rename('TimeNative')],
    axis=1)
df_sorting_kernel_duration['TimeRatio'] = df_sorting_kernel_duration['TimeGo'] / df_sorting_kernel_duration['TimeNative']

#pd.merge(
#    df_go_sorting_kernel_duration,
#    df_native_sorting_kernel_duration,
#    on='Operation',
#    how='inner',
#    suffixes=('Go', 'Native'),
#).drop(columns='Operation')

df_addall_kernel_duration = pd.concat(
    [df_go_addall_kernel_duration['Time'].rename('TimeGo'),
    df_native_addall_kernel_duration['Time'].rename('TimeNative')],
    axis=1)
df_addall_kernel_duration['TimeRatio'] = df_addall_kernel_duration['TimeGo'] / df_addall_kernel_duration['TimeNative']

#df_addall_kernel_duration = pd.merge(
#    df_go_addall_kernel_duration,
#    df_native_addall_kernel_duration,
#    on='Operation',
#    how='inner',
#    suffixes=('Go', 'Native'),
#).drop(columns='Operation')
#df_addall_kernel_duration['TimeRatio'] = df_addall_kernel_duration['TimeGo'] / df_addall_kernel_duration['TimeNative']

def scatter_plot_native_go_relation(ax: plt.Axes, *dfs: tuple[pd.DataFrame, str], x: str, y: str, x_label: str, y_label: str, title: str):
    for color, (df, label) in zip(generate_unique_colors(len(dfs)), dfs):
        df.plot.scatter(x=x, y=y, ax=ax, s=20, color=color, label=label)
    
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    
    ax.grid(axis='both',c='0.9')
    ax.set_axisbelow(True)

labels = {
    #'title': 'Average kernel execution duration',
    'title': 'Average kernel call duration',
    'x_label': 'CUDA C++ (ms)',
    'y_label': 'Go wrapper (ms)',
    'x': 'TimeNative',
    'y': 'TimeGo',
}

def new_scatter_relation_figure(**kwargs) -> tuple[plt.Figure, plt.Axes]:
    (fig, ax) = plt.subplots(**kwargs)
    return fig, ax

figure, ax = new_scatter_relation_figure(figsize=(9,6))
ax.set_yscale('log')
ax.set_xscale('log')

scatter_plot_native_go_relation(
    ax,
    (df_kernel_call_means, 'Sobel'),
    (df_sorting_kernel_call_means, 'Sorting'),
    (df_addall_kernel_call_means, 'Addition'),
#    (df_kernel_duration_means, 'Sobel'),
#    (df_sorting_kernel_duration, 'Sorting'),
#    (df_addall_kernel_duration, 'Addition'),
    **labels)

xbounds = list(ax.get_xlim())
x = np.arange(xbounds[0], xbounds[1], 1)
y = x
ax.plot(x, y, ls='--', color='.3', zorder=-100)
#ax.axis('equal')

figure.tight_layout()
figure.show()
