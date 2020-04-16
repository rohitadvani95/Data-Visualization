import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV



# Import packages to do the classifying


import csv

with open(r'C:\INSERTFILEPATH.csv', 'rU') as infile:
  
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

print(infile)

rawdata = pd.read_csv('C:INSERTFILEPATH.csv', skiprows=0)
#rawdata.drop(["GNPS"], axis = 1, inplace = True)

#print(rawdata)

groupA = pd.DataFrame(data, columns = ['MW', 'SA', 'LogP'])
groupB = data['SA']
groupC = data['LogP']
groupD = data['Skin']

df = pd.DataFrame(data, columns = ['MW', 'SA', 'LogP'])
#df = pd.DataFrame(data, columns = ['MW'])
print(df)

answers = pd.DataFrame(data, columns = ['Skin'])
#print(answers)


#A = [] 

#B = pd.DataFrame(data, columns = ['MW', 'SA'])

#print(B)



#C = 0

D = []

for data['Skin'] in groupD:
    D.append(float(int(data['Skin'])))
print(D)

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        X, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, X, y))

#X = sc.fit_transform(X)
 
#y = sc.fit_transform(y)

array1 = np.asarray(df, dtype=np.float64, order=None)
print(type(array1))
print(array1)
print(len(array1))

print(array1.shape)
array2 = np.asarray(D, dtype=np.int64, order=None)
print(type(array2))
print(array2)
print(len(array2))

print(array2.shape)

iris = load_iris()

X = array1
#print(type(iris.data))
#print(iris.data)
#print(len(iris.data))

y = array2


#print(type(iris.target))
#print(iris.target)

#print(iris.target.shape)
#print(len(iris.target))

# Dataset for decision function visualization: we only keep the first two
# features in X and sub-sample the dataset to keep only 2 classes and
# make it a binary classification problem.

X_2d = X

X_2d = X[:, :2]
#X_2d = X_2d[y > 0]

y_2d = y
#y_2d = y[y > 0]
#y_2d -= 1

print(X_2d)
print(X_2d.shape)

print(y_2d)
print(y_2d.shape)

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)


#print(X)


#print(X_2d)
#print(X_2d.shape)

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 13)
print(C_range)
gamma_range = np.logspace(-9, 3, 13)
print(gamma_range)
param_grid = dict(gamma=gamma_range, C=C_range)
print(param_grid)
cv = StratifiedShuffleSplit(n_splits=3, test_size=0.18, random_state=10)
print(cv)

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)

print(X.shape[1])
grid.fit(X, y)
print(X.shape[1])
#print(grid)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

# #############################################################################
# Visualization
#
# draw visualization of parameter effects

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()




