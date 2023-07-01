# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:17:31 2023

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
import plotly.figure_factory as ff
from plotly.figure_factory import create_dendrogram

#set renderer type
pio.renderers.default= 'svg'

# Import packages to do the classifying

import csv

with open(r'C:\Users\thetr\OneDrive\Documents\Python Scripts\FILEPATHHERE', 'rU') as infile:
  
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

print(infile)

rawdata = pd.read_csv(r'C:\Users\thetr\OneDrive\Documents\Python Scripts\FILEPATHHERE', skiprows=0)

print(rawdata)

df = pd.DataFrame(data, columns = ['Description',	'Number of Peptides', 'Kc-b_1_SUM','Kc-b_2_SUM','Kc-b_3_SUM','Kc-b_4_SUM','Kc-b_5_SUM','Kc-b_6_SUM'])
                 
result = df.sort_values(by=['Kc-b_1_SUM','Kc-b_2_SUM','Kc-b_3_SUM','Kc-b_4_SUM','Kc-b_5_SUM','Kc-b_6_SUM'])

df2 = result.drop(['Description', 'Number of Peptides'], axis = 1)
df2.infer_objects().dtypes

array1 = np.asarray(df2, dtype=np.float32, order=None)

testrows2 = array1[0:50]

print(testrows2)
print(testrows2.shape)

Desc = result['Description'].str.partition('OS=Mus')

edit1 = Desc.loc[:220, [0]]
names = edit1.values.tolist()
print(names)

edit1 = Desc.reset_index(drop=True)

dendro = ff.create_dendrogram(testrows2, orientation='left', labels= names)
dendro.update_layout({'width':2000, 'height':900}) 
dendro.show()
