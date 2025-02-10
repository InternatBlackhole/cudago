#!/usr/bin/env -S python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import os.path as path
import numpy as np
from sklearn.linear_model import LinearRegression
import colorsys

pathToBench = "../results/"

def load_csv(file):
    return pd.read_csv(file, delimiter=';', header='infer')

def load_from_csv(dir, selector):
    df = pd.DataFrame()
    for file in selector:
        dfi = load_csv(path.join(dir, file))
        df = pd.concat([df, dfi]) #ignore_index=True
    return df.reset_index(drop=True)

def generate_unique_colors(n):
    """
    Generate n distinct colors using HSV color space.
    """
    colors = []
    for i in range(n):
        # Use golden ratio to spread hues evenly
        hue = i * 0.618033988749895 % 1
        # Keep saturation and value high for vibrant, distinct colors
        saturation = 0.9
        value = 0.95
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors

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

def scatter_plot_native_go_relation(ax: plt.Axes, *dfs: tuple[pd.DataFrame, str], x: str, y: str, x_label: str, y_label: str, title: str):

    for i, (color, (df, label)) in enumerate(zip(generate_unique_colors(len(dfs)), dfs)):
        first = df.loc[df.index.min()].to_frame().transpose()
        first.plot.scatter(x=x, y=y, ax=ax, s=30, color=color, label='Prvo ' + label.lower(), marker='s')
        df = df.drop(df.index.min())
        df.plot.scatter(x=x, y=y, ax=ax, s=25, color=color, label=label, marker='o')
    
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    
    ax.grid(axis='both',c='0.9')
    ax.set_axisbelow(True)

labels = {
    'title': 'Comparison between precompile and no-precompile',
    'x_label': 'Precompile (\u03BCs)',
    'y_label': 'No-precompile (\u03BCs)',
    'x': 'Time_precomp',
    'y': 'Time_noprecomp',
}

figure, ax = plt.subplots(figsize=(9,6))
ax.set_xscale('log')
ax.set_yscale('log')

#scatter here!!!
scatter_plot_native_go_relation(
    ax,
    (df_mean_kernel_durs, 'Kernel duration'), # in microseconds
    (df_mean_call_durs * 1000, 'Kernel call duration'), # in milliseconds
    **labels
)                         

xbounds = list(ax.get_xlim())
x = np.linspace(xbounds[0], xbounds[1], 500)
y = x
ax.plot(x, y, ls='--', color='.3', zorder=-100)

ax.grid(axis='both',c='0.9')
ax.set_axisbelow(True)

figure.tight_layout()
figure.show()
