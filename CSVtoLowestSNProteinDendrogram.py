# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:35:40 2023

@author: thetr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:31:53 2020

@author: kirby
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

pio.renderers.default= 'svg'

# Import packages to do the classifying


import csv
#make it easy to append all column data from a csv
with open(r'C:\Users\thetr\OneDrive\Documents\Python Scripts\FOLDERHERE\FILENAMEHERE.csv', 'rU') as infile:
  
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

print(infile)

rawdata = pd.read_csv(r'C:\Users\thetr\OneDrive\Documents\Python Scripts\FOLDERHERE\FILENAMEHERE.csv', skiprows=0)

print(rawdata)

df = pd.DataFrame(data, columns = ['Description',	'Number of Peptides', 'SW620-dTAG-13_1_SUM',	'SW620-dTAG-13_2_SUM',	'SW620-dTAG-13_3_SUM'])

df2 = df.drop(['Description', 'Number of Peptides'], axis = 1)
df2.infer_objects().dtypes


result = df2.sort_values(by=['SW620-dTAG-13_1_SUM','SW620-dTAG-13_2_SUM','SW620-dTAG-13_3_SUM'])

print(result)
print(type(result))

array1 = np.asarray(result, dtype=np.float32, order=None)

testrows2 = array1[0:50]
print(testrows2)
print(testrows2.shape)

print(np.count_nonzero(testrows2 < 0, axis=1))

print(type(testrows2))

Desc = df['Description'].to_list()
    
Description = Desc[0:50]
print(Description)
print(type(Description))
names = Description

dendro = ff.create_dendrogram(testrows2, color_threshold=500, orientation='left', labels=names)
dendro.update_layout({'width':2000, 'height':900}) 
dendro.show()


