#!/usr/bin/env -S python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import os.path as path
import numpy as np
from sklearn.linear_model import LinearRegression
import colorsys

pathToBench = "../bench/"

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

op_to_friendy_name = {
    "DevMallocNoReg": "Device malloc (no register)",
    "DevMallocYesReg": "Device malloc (register)",
    "HostMallocNormal": "Host malloc (malloc)",
    "HostMallocCUDA": "Host malloc (cudaHostAlloc)",
    "MemcpyFromDeviceNoReg": "Memcpy from device (no register)",
    "MemcpyFromDeviceYesReg": "Memcpy from device (register)",
    "MemcpyToDeviceNoReg": "Memcpy to device (no register)",
    "MemcpyToDeviceYesReg": "Memcpy to device (register)",
}

#op_to_friendy_name_slo = {
#    "DevMallocNoReg": "Dodelitev na napravi (brez registracije)",
#    "DevMallocYesReg": "Dodelitev na napravi (z registracijo)",
#    "HostMallocNormal": "Dodelitev na gostitelju (malloc)",
#    "HostMallocCUDA": "Dodelitev na gostitelju (cudaHostAlloc)",
#    "MemcpyFromDeviceNoReg": "Prenos iz naprave (brez registracije)",
#    "MemcpyFromDeviceYesReg": "Prenos iz naprave (z registracijo)",
#    "MemcpyToDeviceNoReg": "Prenos na napravo (brez registracije)",
#    "MemcpyToDeviceYesReg": "Prenos na napravo (z registracijo)",
#}

go_dir = path.join(pathToBench, 'go')
native_dir = path.join(pathToBench, 'native')

df_go = load_from_csv(go_dir, filter(lambda x: x.endswith('.csv'), os.listdir(go_dir)))
df_native = load_from_csv(native_dir, filter(lambda x: x.endswith('.csv'), os.listdir(native_dir)))

df_go_grouping = df_go.groupby(by=['Operation'])
df_go_means = df_go_grouping.mean()

df_native_grouping = df_native.groupby(by=['Operation'])
df_native_means = df_native_grouping.mean()

df_means = pd.merge(
    left=df_go_means,
    right=df_native_means,
    on='Operation',
    how='inner',
    suffixes=('Go', 'Native')
)

df_means_all =  pd.concat(
    [df_go.rename(columns={'Time':'TimeGo'}), 
    df_native.rename(columns={'Time': 'TimeNative'}).drop(columns='Operation')],
    axis=1
).set_index('Operation')

def scatter_plot_native_go_relation(ax: plt.Axes, *dfs: tuple[pd.DataFrame, str], x: str, y: str, x_label: str, y_label: str, title: str):
    for i, (color, (df, label)) in enumerate(zip(generate_unique_colors(len(dfs)), dfs)):
        df.plot.scatter(x=x, y=y, ax=ax, s=25, color=color, label=label, marker='o' if len(dfs) // 2 > i else 's')
    
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    
    ax.grid(axis='both',c='0.9')
    ax.set_axisbelow(True)

labels = {
    'title': 'Average time of memory operations',
    'x_label': 'CUDA C++ (ms)',
    'y_label': 'Go wrapper (ms)',
    'x': 'TimeNative',
    'y': 'TimeGo',
}

figure, ax = plt.subplots(figsize=(9,6))
ax.set_xscale('log')
ax.set_yscale('log')

def get_dfs(df: pd.DataFrame):
    dfs = []
    for op in df.index.unique():
        res = df.loc[op]
        #print(res)
        match type(res):
            case pd.Series:
                res = res.to_frame().transpose()
            case pd.DataFrame:
                pass
        dfs.append((res / 1000, op_to_friendy_name[op]))
    return dfs

scatter_plot_native_go_relation(
    ax,
    *get_dfs(df_means_all),
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
