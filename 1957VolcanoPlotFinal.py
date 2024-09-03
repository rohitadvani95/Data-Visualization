#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:17:19 2024

@author: rohitadvani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:08:35 2024

@author: rohitadvani
"""

import pandas as pd
import numpy as np
import scipy

import string
import plotly
import dash_bio

from scipy import stats
from sklearn.datasets import make_blobs

import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

import plotly.io as pio
pio.kaleido.scope.mathjax = None
pio.kaleido.scope.chromium_args += ("--single-process",) 

import plotly.express as px 
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.figure_factory import create_dendrogram

pio.renderers.default= 'png'

import csv

rawdata = pd.read_csv(r'/Users/rohitadvani/Downloads/1957_P1-3_deliverables/P1/1957_P1_protein.csv', header=None, skiprows=0)

df = rawdata.iloc[0:9092, 0:41]
df2 = df.drop(df.iloc[:, 0:4],axis=1)
df2.infer_objects().dtypes

WTC = df.iloc[1:,29:35].astype(float)
WTV = df.iloc[1:,23:29].astype(float)
KC = df.iloc[1:,35:41].astype(float)

WTCsmplstd = WTC.std(axis=1, numeric_only=True)
WTVsmplstd = WTV.std(axis=1, numeric_only=True)
KCsmplstd = KC.std(axis=1, numeric_only=True)


WTC = WTC.assign(mean=WTC.mean(axis=1))
WTV = WTV.assign(mean=WTV.mean(axis=1))
KC = KC.assign(mean=KC.mean(axis=1))

WTC.infer_objects().dtypes
WTV.infer_objects().dtypes
KC.infer_objects().dtypes

WTCavg = (WTC['mean'])
WTVavg = (WTV['mean'])
KCavg = (KC['mean'])

WTClog2 = np.log2(WTCavg) 
WTVlog2 = np.log2(WTVavg)
KClog2 = np.log2(KCavg)

NOP = df.iloc[1:,3:4]
NOParray = np.asarray(NOP, dtype=np.float64, order=None)
print(type(NOParray))

nobs1 = NOParray
nobs2 = nobs1

print(nobs2)
print(nobs2.shape)

mean1 = WTV['mean'].astype(float)
mean2 = WTC['mean'].astype(float)

mean1 = np.asarray(mean1, dtype=np.float64, order=None)
mean1 = mean1[~np.isnan(mean1).all(axis=0)]
print(mean1)

mean2= np.asarray(mean2, dtype=np.float64, order=None)
mean2 = mean2[~np.isnan(mean2).all(axis=0)]
print(mean2)


std1 = WTVsmplstd.astype(float)
std2 = WTCsmplstd.astype(float)

std1 = np.asarray(std1, dtype=np.float64, order=None)
print(type(std1))
std2 = np.asarray(std2, dtype=np.float64, order=None)
print(type(std2))

ttest_output = scipy.stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)

simp_array = np.array(ttest_output, dtype=np.float64 , copy=True, ndmin=1)
print(simp_array[1])

iso_pvals = simp_array[1]

iso_pvals = np.nan_to_num(iso_pvals, nan=0)

#iso_pvals = iso_pvals[~np.isnan(iso_pvals).all(axis=1)]
print(iso_pvals)

Pval_negative_log10 = np.log10(iso_pvals) * (-1)
#KCpvallog10 = np.log10(KCPVALS) * (-1)


fig = go.Figure()
trace1 = go.Scatter(x=WTVlog2,
                    y=Pval_negative_log10,
                    mode='markers',
                    name='KC')

fig.add_trace(trace1)

fig.update_layout(title='Volcano plot WTV')
fig.show()

fig.write_image('fig.png', engine='kaleido')