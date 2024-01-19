# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:08:36 2023

@author: thetr
"""

import pandas as pd
import numpy as np

import string
import plotly
import dash_bio

from sklearn.datasets import make_blobs

import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

import plotly.io as pio
import plotly.express as px 
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.figure_factory import create_dendrogram

pio.renderers.default= 'png'

import csv

rawdata = pd.read_csv(r'C:\Users\thetr\OneDrive\Documents\Python Scripts\1957_P1-3_deliverables\P1\1957_P1_protein.csv', header=None, skiprows=0)

df = rawdata.iloc[0:9092, 0:41]
df2 = df.drop(df.iloc[:, 0:4],axis=1)
df2.infer_objects().dtypes

WTC = df.iloc[1:,29:35].astype(float)
WTV = df.iloc[1:,23:29].astype(float)
KC = df.iloc[1:,35:41].astype(float)

WTC = WTC.assign(mean=WTC.mean(axis=1))
WTV = WTV.assign(mean=WTV.mean(axis=1))
KC = KC.assign(mean=KC.mean(axis=1))

WTV.infer_objects().dtypes
WTC.infer_objects().dtypes
KC.infer_objects().dtypes

WTVlog2 = np.log2(WTV['mean'])
WTClog2 = np.log2(WTC['mean'])
KClog2 = np.log2(KC['mean'])

print(WTVlog2)
print(WTClog2)
print(KClog2)

WTVlog10 = np.log10(WTV['mean']) * (-1)
WTClog10 = np.log10(WTC['mean']) * (-1)
KClog10 = np.log10(KC['mean']) * (-1)

WTVoverWTC = (WTV['mean']/WTC['mean'])
KCoverWTC = (KC['mean']/WTC['mean'])

fig = go.Figure()
trace1 = go.Scatter(x=WTVlog2,
                    y=WTClog10,
                    mode='markers',
                    name='WTV',
                    hovertext=list(df.index))
trace2 = go.Scatter(x=KClog2,
                    y=WTClog10,
                    mode='markers',
                    name='KC',
                    hovertext=list(df.index))

fig.add_trace(trace1)
fig.add_trace(trace2)

fig.update_layout(title='Volcano plot Knockout vs. WTV')
fig.show()


