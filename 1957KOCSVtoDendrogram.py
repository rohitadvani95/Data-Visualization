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

pio.renderers.default= 'svg'

# Import packages to do the classifying


import csv

with open(r'C:\Users\thetr\OneDrive\Documents\Python Scripts\1957_P1-3_deliverables\P1\1957_P1_protein.csv', 'rU') as infile:
  
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

print(infile)

rawdata = pd.read_csv(r'C:\Users\thetr\OneDrive\Documents\Python Scripts\1957_P1-3_deliverables\P1\1957_P1_protein.csv', skiprows=0)

print(rawdata)

#df = pd.DataFrame(data, columns = ['Description',	'Number of Peptides',	'SW620-dTAG-13_1_SUM',	'SW620-dTAG-13_2_SUM',	'SW620-dTAG-13_3_SUM',	'SW620+dTAG-13_1_SUM',	'SW620+dTAG-13_2_SUM',	'SW620+dTAG-13_3_SUM',	'SW48+shCtrl_1_SUM',	'SW48+shCtrl_2_SUM',	'SW48+shCtrl_3_SUM',	'SW48+shW_1_SUM',	'SW48+shW_2_SUM',	'SW48+shW_3_SUM'])
df = pd.DataFrame(data, columns = ['Description',	'Number of Peptides', 'Kc-b_1_SUM','Kc-b_2_SUM','Kc-b_3_SUM','Kc-b_4_SUM','Kc-b_5_SUM','Kc-b_6_SUM'])
                 
#df = pd.DataFrame(data, columns = ['Description', 'SW620-dTAG-13_1',	'SW620-dTAG-13_2',	'SW620-dTAG-13_3',	'SW620+dTAG-13_1',	'SW620+dTAG-13_2',	'SW620+dTAG-13_3',	'SW48+shCtrl_1',	'SW48+shCtrl_2',	'SW48+shCtrl_3',	'SW48+shW_1',	'SW48+shW_2',	'SW48+shW_3'])
#print(df)

#df2 = pd.DataFrame(data,
                  #columns= cols, 
                  #).astype(float)
#print(df)
#print(df.shape)



#print(df2.to_csv(header=None,index=False)
#truncdf2 = df2.drop([0,1])
#print(df2)
#print(truncdf2)
#print(truncdf2.dtypes)

#result = df2.groupby('SW620-dTAG-13_1_SUM','SW620-dTAG-13_2_SUM')['SW620-dTAG-13_3_SUM'].aggregate(['min','max'])

result = df.sort_values(by=['Kc-b_1_SUM','Kc-b_2_SUM','Kc-b_3_SUM','Kc-b_4_SUM','Kc-b_5_SUM','Kc-b_6_SUM'])

#result.sort_values('Max')
#print(result)
#print(type(result))

df2 = result.drop(['Description', 'Number of Peptides'], axis = 1)

df2.infer_objects().dtypes




#print(df1)
#print(df1.shape)
#print(type(df1))

#df1.infer_objects().dtypes

array1 = np.asarray(df2, dtype=np.float32, order=None)

#print(array1)
#print(array1.shape)

testrows2 = array1[0:50]

print(testrows2)
print(testrows2.shape)

#Desc = result['Description']

Desc = result['Description'].str.partition('OS=Mus')

edit1 = Desc.loc[:220, [0]]
names = edit1.values.tolist()
print(names)
#print(names.shape)


edit1 = Desc.reset_index(drop=True)



#print(Desc.loc[:220, [0]].shape)

#print(Desc.shape)
#print(type(Desc))


#EditedNames = Desc[0:50,:0]
#print(EditedNames)

#print(type(EditedNames))

#print(df1)
#print(df1.shape)

#names = df1.drop(columns=['1', '2'])

#print(names)
#print(names.shape)
#for columns in cols_to_change:
 #   df[columns] = df[columns].str.replace('[Solute]', '')

#names = EditedNames.to_list()

#Desc = df['Description'].to_string()
#DescEdit = Desc.replace('OS=Homo sapiens', '', regex=False)

##Description = Desc[0:50].to_list()




#fig = ff.create_dendrogram(testrows2, color_threshold=1.5)
#names = Description
#fig = ff.create_dendrogram(testrows2, orientation='left', labels=names)

#fig.update_layout(width=800, height=900)
#fig.show()



dendro = ff.create_dendrogram(testrows2, orientation='left', labels= names)
dendro.update_layout({'width':2000, 'height':900}) 
dendro.show()


