# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

pio.renderers.default= 'png'

# Import packages to do the classifying



import csv
#make it easy to append all column data from a csv
with open(r'/Users/rohitadvani/Downloads/1957_P1-3_deliverables/P1/1957_P1_protein.csv', 'rU') as infile:
  
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

#print(infile)

rawdata = pd.read_csv(r'/Users/rohitadvani/Downloads/1957_P1-3_deliverables/P1/1957_P1_protein.csv', header=None, skiprows=0)

#print(rawdata)

df = pd.DataFrame(data, columns = ['Description','Number of Peptides','Kc-b_1_SUM','Kc-b_2_SUM','Kc-b_3_SUM','Kc-b_4_SUM','Kc-b_5_SUM','Kc-b_6_SUM'], dtype=float)

df2 = df.drop(['Description', 'Number of Peptides'], axis = 1)
df2.infer_objects().dtypes


result = df2.sort_values(by=['Kc-b_1_SUM','Kc-b_2_SUM','Kc-b_3_SUM','Kc-b_4_SUM','Kc-b_5_SUM','Kc-b_6_SUM'])

print(result)
print(type(result))


#result = result.astype(float)
#print(result.dtypes)

result = result.assign(mean=result.mean(axis=1))

print(result)
print(result.shape)


#print(result.mean(axis=1))



ctrldf = pd.DataFrame(data, columns = ['Description','Number of Peptides','Wc-b_1_SUM','Wc-b_2_SUM','Wc-b_3_SUM','Wc-b_5_SUM','Wc-b_6_SUM'], dtype=float)

print(ctrldf)

ctrldf = ctrldf.drop(['Description', 'Number of Peptides'], axis = 1)
ctrldf.infer_objects().dtypes

print(ctrldf)

ctrl1 = ctrldf.sort_values(by=['Wc-b_1_SUM','Wc-b_2_SUM','Wc-b_3_SUM','Wc-b_5_SUM','Wc-b_6_SUM'])

print(ctrl1)


ctrl1 = ctrl1.assign(mean=ctrl1.mean(axis=1))

print(ctrl1)

result['ratio'] = (result['mean']/ ctrl1['mean']*100).round(2)
result['log'] = np.log2(result['ratio'])
result = result.sort_values(by=['log'])

print(result)

print(result['log'])

result2 = result['log']

array1 = np.asarray(result2, dtype=np.float32, order=None)

print(array1)
print(type(array1))
print(array1.shape)

#df3 = result['mean'] /= ctrl1['mean']

testrows2 = array1[0:50]
print(testrows2)
print(testrows2.shape)

#print(np.count_nonzero(testrows2 < 0, axis=1))

print(type(testrows2))

df = df.reindex(result.index)

print(df)

#Desc = df['Description'].to_list()


Desc = df['Description'].str.partition('OS=Mus')
#Desc = Desc.drop(['1','2',], axis=1)
 
print(Desc)
#Description = Desc[0:50]
#print(Description)
#print(type(Description))
#names = Description

#dendro = ff.create_dendrogram(testrows2, color_threshold=500, orientation='left', labels=names)
#dendro.update_layout({'width':2000, 'height':900}) 
#dendro.show()