import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('result_log.csv', header=None, names=['version', 'nbcore', 'num_steps', 'runtime'], dtype={
    'version': str,
    'nbcore': int,
    'num_steps': int,
    'runtime': float
})

color_num_steps = {1000000: "blue", 100000000: "red", 10000000000: "green", 1000000000000: "black"}

for num_steps in df['num_steps']:
    df_plot = df[(df['num_steps'] == int(num_steps))]
    df_plot = df_plot[df_plot['version'] == "atomic"]

    mean_stats = df_plot.groupby(['num_steps', 'version', 'nbcore']).mean().reset_index()

    plt.plot(mean_stats['nbcore'], mean_stats['runtime'], linestyle="solid", color=color_num_steps[num_steps])
    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(df_plot['nbcore'], df_plot['runtime'], color=color_num_steps[num_steps])

    df_plot = df[(df['num_steps'] == num_steps) & (df['version'] == "reduce")]
    mean_stats = df_plot.groupby(['num_steps', 'version', 'nbcore']).mean().reset_index()

    plt.plot(mean_stats['nbcore'], mean_stats['runtime'], linestyle="dashed", color=color_num_steps[num_steps])
    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(df_plot['nbcore'], df_plot['runtime'], color=color_num_steps[num_steps])

plt.legend()
plt.show()