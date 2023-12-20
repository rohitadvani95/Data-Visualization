# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:08:36 2023

@author: thetr
"""

import pandas as pd
import numpy as np

import string
import plotly

from sklearn.datasets import make_blobs

import seaborn as sns
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

WTV = np.log2(WTV)
WTC = np.log2(WTC)
KC = np.log2(KC)

WTVoverWTC = (WTV['mean']/WTC['mean'])
KCoverWTC = (KC['mean']/WTC['mean'])

dftoPlot = pd.DataFrame(columns = ['Treatment 1 Log 2 Dev. from ctrl', 'KC Log2 Dev. from ctrl'])

dftoPlot['Treatment 1 Log 2 Dev. from ctrl'] = WTVoverWTC
dftoPlot['KC Log2 Dev. from ctrl'] = KCoverWTC
#dftoPlot['Cat'] = ''

df4 = dftoPlot.melt(value_vars =['KC Log2 Dev. from ctrl'])
df3 = dftoPlot.melt(value_vars =['Treatment 1 Log 2 Dev. from ctrl'])
df5 = pd.concat([df3,df4])

sns.violinplot(data=df5, x="variable", y="value")
