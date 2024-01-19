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

dftoPlot = df.iloc[1:,23:41].astype(float)
dftoPlot.infer_objects().dtypes

#sns.color_palette("coolwarm")
sns.clustermap(dftoPlot, 
               cmap ="coolwarm", 
               method='median', 
               metric='correlation', 
               z_score=None, 
               standard_scale=None, 
               figsize=(30, 30), 
               cbar_kws=None, 
               row_cluster=True, 
               col_cluster=True, 
               row_linkage=None, 
               col_linkage=None, 
               dendrogram_ratio=0.2, 
               colors_ratio=2)