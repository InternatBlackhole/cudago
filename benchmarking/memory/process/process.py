#!/usr/bin/env -S python3

import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as path
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

op_to_friendy_name_slo = {
    #"DevMallocNoReg": "Dodelitev na napravi (brez registracije)",
    #"DevMallocYesReg": "Dodelitev na napravi (z registracijo)",
    #"HostMallocNormal": "Dodelitev na gostitelju (malloc)",
    #"HostMallocCUDA": "Dodelitev na gostitelju (cudaHostAlloc)",
    #"MemcpyFromDeviceNoReg": "Prenos iz naprave (brez registracije)",
    #"MemcpyFromDeviceYesReg": "Prenos iz naprave (z registracijo)",
    #"MemcpyToDeviceNoReg": "Prenos na napravo (brez registracije)",
    #"MemcpyToDeviceYesReg": "Prenos na napravo (z registracijo)",
    "DevMallocNoReg": "brez registracije",
    "DevMallocYesReg": "z registracijo",
    "HostMallocNormal": "malloc",
    "HostMallocCUDA": "cudaHostAlloc",
    "MemcpyFromDeviceNoReg": "Iz naprave (brez registracije)",
    "MemcpyFromDeviceYesReg": "Iz naprave (z registracijo)",
    "MemcpyToDeviceNoReg": "Na napravo (brez registracije)",
    "MemcpyToDeviceYesReg": "Na napravo (z registracijo)",
}

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

def new_line_after_n_words(s: str, n: int):
    words = s.split()
    return '\n'.join(' '.join(words[i:i+n]) for i in range(0, len(words), n))

figure = plt.figure(figsize=(9,6), layout='tight')

kwargs = {
    'xlabel': '',
    'ylabel': 'Čas (ms)',
    'rot': 0
}

mainGrid = figure.subplot_mosaic([
    ['dev_allocs', 'host_allocs'],
    ['transfers', 'transfers']
]
)

#allocsGrid = mainGrid[0].subgridspec(1, 2)
axAllocDev = mainGrid['dev_allocs'] #figure.add_subplot(allocsGrid[0])
axAllocHost = mainGrid['host_allocs'] #figure.add_subplot(allocsGrid[1])

#transfersGrid = mainGrid[1].subgridspec(1, 1)
axTransfers = mainGrid['transfers'] #figure.add_subplot(transfersGrid[0])

column_rename = {'TimeNative': 'CUDA C++', 'TimeGo': 'Ovojnica go'}
index_rename = {k: new_line_after_n_words(v, 2) for k, v in op_to_friendy_name_slo.items()}

def getterRename(df: pd.DataFrame, contains: str, regex = False) -> pd.DataFrame:
    return df.loc[df.index.str.contains(contains, regex=regex)].rename(columns=column_rename, index=index_rename)

df = df_means / 1000 # ms

df_device_alloc = getterRename(df, 'DevMalloc')
df_host_alloc = getterRename(df, 'HostMalloc')
df_mem_transfers = getterRename(df, 'Memcpy')

df_device_alloc.plot.bar(ax=axAllocDev, title='Povprečni čas dodelitve pomnilnika na napravi', legend=False, **kwargs)
df_host_alloc.plot.bar(ax=axAllocHost, title='Povprečni čas dodelitve pomnilnika na gostitelju', legend=True, **kwargs)
df_mem_transfers.plot.bar(ax=axTransfers, title='Povprečni čas prenosa podatkov', logy=True, legend=True, **kwargs)

formatter = lambda x, _: f'{int(x) if x == int(x) else round(x, 3)}'

#axAllocDev.legend(loc='lower right')
#axAllocHost.legend(loc='lower left')

axAllocDev.yaxis.set_major_formatter(plt.FuncFormatter(formatter))
axAllocDev.yaxis.set_minor_formatter(plt.FuncFormatter(formatter))

axAllocHost.yaxis.set_major_formatter(plt.FuncFormatter(formatter))
#axAllocHost.yaxis.set_minor_formatter(plt.FuncFormatter(formatter))

axTransfers.yaxis.set_major_formatter(plt.FuncFormatter(formatter))
axTransfers.yaxis.set_minor_formatter(plt.FuncFormatter(formatter))

#figure.tight_layout()
figure.show()