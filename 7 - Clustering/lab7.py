#!/usr/bin/env python
# coding: utf-8

# # imports

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix

import pickle


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[164]:


from sklearn.cluster import DBSCAN


# # data

# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=True)
mnist.target = mnist.target.astype(np.uint8)


# In[3]:


X = mnist["data"]
y = mnist["target"]


# # KMeans

# In[4]:


n_kmeans = [8, 9, 10, 11, 12]


# In[5]:


kmeans_clfs = []
preds = []

for n in range(len(n_kmeans)):
    print(f"n = {n_kmeans[n]}\n")
    clf = KMeans(n_clusters=n_kmeans[n], random_state=42)
    kmeans_clfs.append(clf)
    preds.append(clf.fit_predict(X))


# In[ ]:





# ## silhoutte scores

# In[6]:


s_scores = []


# In[7]:


for clf in kmeans_clfs:
    s_score = silhouette_score(X, clf.labels_)
    s_scores.append(s_score)


# In[ ]:


s_scores


# In[14]:


with open("kmeans_sil.pkl", "wb") as f:
    pickle.dump(s_scores, f, pickle.HIGHEST_PROTOCOL)


# ## confusion matrix

# In[10]:


print(kmeans_clfs[2], preds[2])
confusion_matrix_k10 = confusion_matrix(y, preds[2])
print(confusion_matrix_k10)


# In[9]:


indices_max = np.argmax(confusion_matrix_k10, axis=1)
indices_max


# In[11]:


indices_max_set = set(indices_max)
indices_max_set


# In[12]:


with open("kmeans_argmax.pkl", "wb") as f:
    pickle.dump(indices_max_set, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:





# In[ ]:





# # DBSCAN

# In[49]:


distances = set()


# In[50]:


for i in range(300):
    for j in range(300):
        distances.add(np.linalg.norm(X.loc[i] - X.loc[j]))


# In[70]:


distances_list = list(distances)
distances_list.sort()


# In[72]:


dist = []

for i in range(1, 11):
    dist.append(distances_list[i])


# In[74]:


with open("dist.pkl", "wb") as f:
    pickle.dump(dist, f, pickle.HIGHEST_PROTOCOL)


# In[154]:


eps_list = []


# In[162]:


s = sum(dist[7:]) / 3
s_top = s * 1.1
step = s * 0.04

index = 0
while s < s_top:
    eps_list.append(s)
    s += step


# In[169]:


dbscans = []

for eps in eps_list:
    print(f"eps = {eps}")
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(X)
    dbscans.append(dbscan)
    print("appended")


# In[177]:


dbscan_len_list = []

for dbscan in dbscans:
    dbscan_len_list.append(len(set(dbscan.labels_)))


# In[178]:


with open("dbscan_len.pkl", "wb") as f:
    pickle.dump(dbscan_len_list, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




