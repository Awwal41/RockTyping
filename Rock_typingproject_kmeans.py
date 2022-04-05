#!/usr/bin/env python
# coding: utf-8

# In[1]:


# K Means Cluster Analysis

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


# Importing Dataset
dataset = pd.read_csv(r'C:\Users\bglag\Documents\kisdata.csv')


# In[7]:


dataset.describe()


# In[23]:


new_column_names = {'DEPTH': "depth", 
        'PoroRCA': "porosity",
        'PermRCA': "permeability", 
        'PHIE_2014': "phie", 
        'RQI': "rqi",
        'PHIz': "phiz",
        'FZI': "fzi",
        'Log FZI': "logfzi",
        'Percentile': "percentile", 
        'HFU_RT': "rocktype",
        'HFU_Rock Type': "rocktype"}


# In[28]:


##X=[ "depth", "porosity","permeability","phie", "rqi", "phiz","fzi","logfzi","percentile"]
X = dataset.iloc[:, [3,4]].values


# In[29]:


# Elbow Method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,6), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()


# In[30]:


# fitting kmeans to dataset
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans = kmeans.fit_predict(X)


# In[31]:


# Visualising the clusters
plt.scatter(X[Y_kmeans==0, 0], X[Y_kmeans==0, 1], s=100, c='red', label= 'Cluster 1')
plt.scatter(X[Y_kmeans==1, 0], X[Y_kmeans==1, 1], s=100, c='cyan', label= 'Cluster 2')
plt.scatter(X[Y_kmeans==2, 0], X[Y_kmeans==2, 1], s=100, c='green', label= 'Cluster 3')
plt.scatter(X[Y_kmeans==3, 0], X[Y_kmeans==3, 1], s=100, c='blue', label= 'Cluster 4')
plt.scatter(X[Y_kmeans==4, 0], X[Y_kmeans==4, 1], s=100, c='magenta', label= 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids' )
plt.title('Clusters of rock types')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()


# In[ ]:




