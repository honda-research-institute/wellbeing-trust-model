import pandas as pd
import glob
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats

def t_test(a, b, mode = 'two-sided'):
    a = pd.to_numeric(a)
    b = pd.to_numeric(b)
    print(pg.ttest(a, b,alternative=mode))

def pearson_corr(a, b):
    a = pd.to_numeric(a)
    b = pd.to_numeric(b)
    print(pg.corr(a, b))

def pointbiserial_corr(a, b):
    a = pd.to_numeric(a)
    b = pd.to_numeric(b)
    print(stats.pointbiserialr(a, b))

def spearman_corr(a, b):
    a = pd.to_numeric(a)
    b = pd.to_numeric(b)
    print(stats.spearmanr(a, b))

def mann_whitney(a, b, mode = 'two-sided'):
    a = pd.to_numeric(a)
    b = pd.to_numeric(b)
    print(pg.mwu(a,b,alternative=mode))

def chi_square(data, a, b):
    # a = pd.to_numeric(a)
    # b = pd.to_numeric(b)
    print(pg.chi2_independence(data, x=a, y=b))

def corr_plot(df, df_corr, title='title'):
    plt.figure(figsize=(12, 8))
    for col in df.columns:
        sns.regplot(x=df[df.columns[-1]], y=df[col], line_kws={"color": "r", "alpha": 0.7, "lw": 5})
        plt.savefig(title+col + ".png")
        plt.show()
    plt.figure(figsize=(15, 15))
    sns.heatmap(df_corr, annot=True)
    plt.savefig(title+'heatmap' + ".png")
    plt.show()