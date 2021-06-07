# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:39:34 2021

@author: Lenovo
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#import dataset
dataset = pd.read_csv('heart.csv')

# we used ALL ROWS
dataset = pd.read_csv('heart.csv')
x= dataset.iloc[:,:].values

from sklearn.preprocessing import normalize
data_scaled = normalize(dataset)
data_scaled = pd.DataFrame(data_scaled, columns=dataset.columns)


from sklearn.cluster import KMeans
# here we are using elbow method in order to find optimal clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)

#the cluster is determined when th interia starts to decrease in linear way
plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), inertia,marker='o',color='red')
plt.title(' Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()


# fit k means to dataset
kmeans=KMeans(n_clusters=7, init = 'k-means++', random_state=42)
y_kmeans =kmeans.fit_predict(x)

#visualising the cluster
plt.figure(figsize=(15,30))
plt.title('Heart cluster kmeans')
sns.scatterplot(x[y_kmeans==0,0],x[y_kmeans ==0,1],s=100,color='red',label='cluster1')
sns.scatterplot(x[y_kmeans==1,0],x[y_kmeans ==1,1],s=100, color='green',label='cluster2')
sns.scatterplot(x[y_kmeans==2,0],x[y_kmeans ==2,1],s=100, color='blue',label='cluster3')
sns.scatterplot(x[y_kmeans==3,0],x[y_kmeans ==3,1],s=100, color='yellow',label='cluster4')
sns.scatterplot(x[y_kmeans==4,0],x[y_kmeans ==4,1],s=100, color='cyan',label='cluster5')
sns.scatterplot(x[y_kmeans==5,0],x[y_kmeans ==5,1],s=100, color='purple',label='cluster6')





