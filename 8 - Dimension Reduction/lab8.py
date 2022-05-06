#!/usr/bin/env python
# coding: utf-8

# # imports

# In[2]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

import pickle
import seaborn as sns
from sklearn.decomposition import PCA


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


from sklearn.preprocessing import StandardScaler


# # load data

# In[6]:


bc = load_breast_cancer(as_frame=True)
iris = load_iris(as_frame=True)


# In[7]:


X_bc = bc["data"]
y_bc = bc["target"]


# In[8]:


X_iris = iris["data"]
y_iris = iris["target"]


# In[9]:


# X_bc.info()


# In[10]:


# X_iris.info()


# In[11]:


# X_dbc.sample(4)


# In[12]:


# X_iris.sample(4)


# # Breast Cancer

# ## PCA

# In[13]:


pca_bc = PCA(n_components=.9)


# In[14]:


X_bc_pca = pca_bc.fit_transform(X_bc)


# In[15]:


print(pca_bc.explained_variance_ratio_, pca_bc.n_components_)


# ## Scaled

# In[16]:


scaler_bc = StandardScaler()


# In[51]:


X_bc_scaled = scaler_bc.fit_transform(X_bc)


# In[18]:


pca_bc_scaled = PCA(n_components=.9)


# In[19]:


X_bc_pca_scaled = pca_bc_scaled.fit_transform(X_bc_scaled)


# In[20]:


# print(f"""
# {pca_bc_scaled.explained_variance_ratio_}
# {pca_bc_scaled.n_components_}
# {pca_bc_scaled.components_}
# """)


# ### Wspolczynniki zmiennosci

# In[21]:


list_var_ratio_bc = list(pca_bc_scaled.explained_variance_ratio_)
list_var_ratio_bc


# ### pickle the list

# In[37]:


with open("pca_bc.pkl", "wb") as f:
    pickle.dump(list_var_ratio_bc, f, pickle.HIGHEST_PROTOCOL)


# ## pickle indices

# In[61]:


list_indices_max_bc = pd.DataFrame(pca_bc_scaled.components_).idxmax(axis="columns").tolist()


# In[62]:


with open("idx_bc.pkl", "wb") as f:
    pickle.dump(list_indices_max_bc, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:





# In[ ]:





# # Iris

# ## PCA

# In[23]:


pca_iris = PCA(n_components=.9)


# In[24]:


X_iris_pca = pca_iris.fit_transform(X_iris)


# In[25]:


print(pca_iris.explained_variance_ratio_, pca_iris.n_components_)


# ## Scaled

# In[26]:


scaler_iris = StandardScaler()


# In[27]:


X_iris_scaled = scaler_iris.fit_transform(X_iris)


# In[28]:


pca_iris_scaled = PCA(n_components=.9)


# In[29]:


X_iris_pca_scaled = pca_iris_scaled.fit_transform(X_iris_scaled)


# In[30]:


print(pca_iris_scaled.explained_variance_ratio_, pca_iris_scaled.n_components_)


# ### Wspolczynniki zmiennosci

# In[32]:


list_var_ratio_iris = list(pca_iris_scaled.explained_variance_)
list_var_ratio_iris


# ### pickle the list

# In[36]:


with open("pca_ir.pkl", "wb") as f:
    pickle.dump(list_var_ratio_iris, f, pickle.HIGHEST_PROTOCOL)


# ## pickle indices

# In[58]:


list_indices_max_iris = pd.DataFrame(pca_iris_scaled.components_).idxmax(axis="columns").tolist()


# In[63]:


with open("idx_ir.pkl", "wb") as f:
    pickle.dump(list_var_ratio_iris, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




