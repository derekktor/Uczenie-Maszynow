#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)


# In[17]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


# In[3]:


X = mnist.data
y = mnist.target.astype(np.uint8)
print(X.shape, y.shape)


# In[4]:


# # try sorting
# y = y.sort_values(ascending=True)
# X = X.reindex()


# In[5]:


# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[6]:


# testing uniqueness
print(np.unique(y_train))
print(np.unique(y_test))


# In[7]:


# checking only zeros
y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)


# In[8]:


# importing the classifier
from sklearn.linear_model import SGDClassifier


# In[9]:


# feeding the model data
start = time.time()
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)
print(time.time() - start)


# In[10]:


# measuring the accuracy of the clf
start = time.time()
score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
print(time.time() - start)
print(score)


# In[11]:


y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_0, cv=3, n_jobs=-1)
print(y_train_pred)
print(confusion_matrix(y_train_0, y_train_pred))


# In[13]:


import pickle


# In[15]:


accuracies = [sgd_clf.score(X_train, y_train_0), sgd_clf.score(X_test, y_test_0)]
with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(accuracies, f, pickle.HIGHEST_PROTOCOL)


# In[19]:


cross_score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(cross_score, f, pickle.HIGHEST_PROTOCOL)


# In[21]:


# feeding the model data
start = time.time()
sgd_wiel_clf = SGDClassifier(random_state=42,n_jobs=-1)
sgd_wiel_clf.fit(X_train, y_train)
print(time.time() - start)


# In[23]:


y_train_pred = cross_val_predict(sgd_wiel_clf, X_train, y_train, cv=3, n_jobs=-1)
matrix = confusion_matrix(y_train, y_train_pred)
with open('sgd_cmx.pkl', 'wb') as f:
    pickle.dump(matrix, f, pickle.HIGHEST_PROTOCOL)


# In[24]:


with open('sgd_acc.pkl', 'rb') as f:
    print(pickle.load(f))
    
with open('sgd_cva.pkl', 'rb') as f:
    print(pickle.load(f))
    
with open('sgd_cmx.pkl', 'rb') as f:
    print(pickle.load(f))


# In[ ]:




