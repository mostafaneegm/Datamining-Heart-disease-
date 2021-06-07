# -*- coding: utf-8 -*-
"""
Created on Wed May 12 21:09:47 2021

@author: Lenovo
"""



import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift as ms
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# we used only sex, fbs, restecg, exang, and ca
dataset = pd.read_csv('heart.csv')
x= dataset.iloc[:,[1,5,6,8,11]].values

#normlize dataset
from sklearn.preprocessing import normalize
data_scaled = normalize(dataset)
data_scaled = pd.DataFrame(data_scaled, columns=dataset.columns)

#the dendrogram 
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))

# see where the line is intersecting and according to it is the number of the cluster
plt.axhline(y=10, color='r', linestyle='--')


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean',linkage='ward') 
y_hc = hc.fit_predict(x)

#cluster visualization
plt.figure(figsize=(15,7))
plt.scatter(x[y_hc==0,0], x[y_hc==0,1],s =100, c='red',label='Cluster 1')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1],s =100, c='green',label='Cluster 2')
plt.scatter(x[y_hc==2,0], x[y_hc==2,1],s =100, c='blue',label='Cluster 3')


plt.title('Heart cluster Hierarchy')

plt.legend()
plt.show()