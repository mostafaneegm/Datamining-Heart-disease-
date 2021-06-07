# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:56:34 2021

@author: Lenovo
"""
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift as ms
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# we used only trestbps, chol, and thalach
dataset = pd.read_csv('heart.csv')
x= dataset.iloc[:,[3,4,7]].values

#normalize data
from sklearn.preprocessing import normalize
data_scaled = normalize(dataset)
data_scaled = pd.DataFrame(data_scaled, columns=dataset.columns)

#the dendrogram 
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))

# see where the line is intersecting and according to it is the number of the cluster
plt.axhline(y=500, color='r', linestyle='--')

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=2, affinity='euclidean',linkage='ward') 
y_hc = hc.fit_predict(x)

#cluster visualization
plt.figure(figsize=(15,30))
plt.scatter(x[y_hc==0,0], x[y_hc==0,1],s =100, c='red',label='Cluster 1')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1],s =100, c='green',label='Cluster 2')


plt.title('Heart cluster Hierarchy')

plt.legend()
plt.show()