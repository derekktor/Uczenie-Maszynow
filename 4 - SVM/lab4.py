#!/usr/bin/env python
# coding: utf-8

# ### Importing the modules

# In[143]:


import numpy as np
# breast_cancer; iris
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ### Loading the data

# In[144]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
# print(data_breast_cancer["DESCR"])
# data_breast_cancer.target
data_breast_cancer.data[["mean area", "mean smoothness"]]
# data_breast_cancer.frame


# In[151]:


data_iris = datasets.load_iris(as_frame=True)
# print(data_iris["DESCR"])
# data_iris.target
data_iris.data[["petal length (cm)", "petal width (cm)"]]
# data_iris.frame


# ## Breast Cancer

# ### Target features(X), Prediction target(y) 

# In[152]:


X_b = data_breast_cancer.data[["mean area", "mean smoothness"]]
y_b = data_breast_cancer.target


# ### Splitting the data

# In[147]:


X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, test_size=.2)


# ### Creating the models

# In[148]:


svm_b_without_clf = LinearSVC(C=1, loss="hinge", random_state=42, max_iter=1000)

svm_b_with_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42))
])


# ### Fitting the model

# In[153]:


svm_b_without_clf.fit(X_b_train, y_b_train)
svm_b_with_clf.fit(X_b_train, y_b_train)


# ### Calculating accuracy

# In[154]:


from sklearn.metrics import accuracy_score


# In[155]:


b_train_pred_without = svm_b_without_clf.predict(X_b_train)
b_test_pred_without = svm_b_without_clf.predict(X_b_test)

b_train_pred_with = svm_b_with_clf.predict(X_b_train)
b_test_pred_with = svm_b_with_clf.predict(X_b_test)


# In[156]:


acc_train_with = accuracy_score(y_b_train, b_train_pred_with)
acc_test_with = accuracy_score(y_b_test, b_test_pred_with)


# In[157]:


acc_train_without = accuracy_score(y_b_train, b_train_pred_without)
acc_test_without = accuracy_score(y_b_test, b_test_pred_without)


# In[158]:


bc_list = [acc_train_without, acc_test_without, acc_train_with, acc_test_with]
bc_list


# ## Iris

# ### Target features(X), Prediction target(y: Petal [Length | Width])

# In[109]:


X_i = data_iris.data[["petal length (cm)", "petal width (cm)"]]
y_i = (data_iris.target == 2).astype(int)


# ### Splitting the data: {train, test}

# In[110]:


X_i_train, X_i_test, y_i_train, y_i_test = train_test_split(X_i, y_i, test_size=.2)


# ### Creating the models

# In[111]:


svm_i_without_clf = LinearSVC(C=1, loss="hinge", random_state=42, max_iter=10000)

svm_i_with_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42))
])


# ### Fitting the models

# In[112]:


svm_i_without_clf.fit(X_i_train, y_i_train)
svm_i_with_clf.fit(X_i_train, y_i_train)


# In[113]:


i_train_pred_without = svm_i_without_clf.predict(X_i_train)
i_test_pred_without = svm_i_without_clf.predict(X_i_test)

i_train_pred_with = svm_i_with_clf.predict(X_i_train)
i_test_pred_with = svm_i_with_clf.predict(X_i_test)


# In[114]:


acc_i_train_with = accuracy_score(y_i_train, i_train_pred_with)
acc_i_test_with = accuracy_score(y_i_test, i_test_pred_with)

acc_i_train_without = accuracy_score(y_i_train, i_train_pred_without)
acc_i_test_without = accuracy_score(y_i_test, i_test_pred_without)


# In[122]:


i_list = [acc_i_train_without, acc_i_test_without, acc_i_train_with, acc_i_test_with]
i_list


# In[129]:


import pickle


# ## Pickle the results

# In[130]:


with open("bc_acc.pkl", "wb") as f:
    pickle.dump(bc_list, f, pickle.HIGHEST_PROTOCOL)


# In[131]:


with open("iris_acc.pkl", "wb") as f:
    pickle.dump(i_list, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:




