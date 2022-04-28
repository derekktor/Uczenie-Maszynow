#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import pickle


# # Load Data

# In[2]:


dbc = load_breast_cancer(as_frame=True)


# ## Features

# In[3]:


features = ["mean texture", "mean symmetry"]
X = dbc["data"][features]
X_all = dbc["data"]
X_all.head()


# ## Target

# In[4]:


y = dbc["target"]
y.head()


# # Split data

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# In[6]:


X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y, test_size=.2)


# # Classifiers

# In[7]:


tree_clf = DecisionTreeClassifier(random_state=42)
log_reg = LogisticRegression(random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=3)
vsoft_clf = VotingClassifier(
    estimators=[
        ('tree', tree_clf),
        ('log', log_reg),
        ('knn', knn_clf)
    ],
    voting='soft'
)
vhard_clf = VotingClassifier(
    estimators=[
        ('tree', tree_clf),
        ('log', log_reg),
        ('knn', knn_clf)
    ],
    voting='hard'
)

clfs = [tree_clf, log_reg, knn_clf, vsoft_clf, vhard_clf]


# ## fit estimators

# In[8]:


for clf in clfs:
    clf.fit(X_train, y_train)


# ## get accuracy scores

# In[9]:


def get_pred(clf, X_train, X_test):
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    return y_pred_train, y_pred_test


# In[10]:


def get_acc_score(clf, y_train, y_test, y_pred_train, y_pred_test):
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    return acc_train, acc_test


# In[11]:


acc_scores = dict()


# In[12]:


for clf in clfs:
    acc = get_acc_score(clf, y_train, y_test, get_pred(clf, X_train, X_test)[0], get_pred(clf, X_train, X_test)[1])
    acc_scores[f"{clf}"] = (clf.__class__.__name__, acc)

for s in acc_scores:
    print(acc_scores[s])


# ## pickle accuracies

# In[13]:


vote_acc_list = []


# In[14]:


for clf in clfs:
    acc = get_acc_score(clf, y_train, y_test, get_pred(clf, X_train, X_test)[0], get_pred(clf, X_train, X_test)[1])
    vote_acc_list.append(acc)

vote_acc_list


# In[15]:


with open("acc_vote.pkl", "wb") as f:
    pickle.dump(vote_acc_list, f, pickle.HIGHEST_PROTOCOL)


# In[16]:


with open("vote.pkl", "wb") as f:
    pickle.dump(clfs, f, pickle.HIGHEST_PROTOCOL)


# # Bagging, Pasting, AdaBoost...

# In[17]:


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    bootstrap=True, random_state=42, max_features=2
)

bag_clf50 = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=.5, bootstrap=True, random_state=42, max_features=2
)

pas_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    bootstrap=False, random_state=42, max_features=2
)

pas_clf50 = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    max_samples=.5, bootstrap=False, random_state=42, max_features=2
)

frst_clf = RandomForestClassifier(
    n_estimators=30, random_state=42
)

ada_clf = AdaBoostClassifier(n_estimators=30)

grad_clf = GradientBoostingClassifier(n_estimators=30, random_state=42)

clfs2 = [bag_clf, bag_clf50, pas_clf, pas_clf50, frst_clf, ada_clf, grad_clf]


# ## fit classifiers

# In[18]:


for clf in clfs2:
    clf.fit(X_train, y_train)


# ## get accuracy scores

# In[19]:


acc_scores2 = dict()


# In[20]:


for clf in clfs2:
    acc = get_acc_score(clf, y_train, y_test, get_pred(clf, X_train, X_test)[0], get_pred(clf, X_train, X_test)[1])
    acc_scores2[f"{clf}"] = (clf.__class__.__name__, acc)

for s in acc_scores:
    print(acc_scores[s])


# ## pickle accuracies

# In[21]:


bag_acc_list = []


# In[22]:


for clf in clfs2:
    acc = get_acc_score(clf, y_train, y_test, get_pred(clf, X_train, X_test)[0], get_pred(clf, X_train, X_test)[1])
    bag_acc_list.append(acc)

bag_acc_list


# In[23]:


with open("acc_bag.pkl", "wb") as f:
    pickle.dump(bag_acc_list, f, pickle.HIGHEST_PROTOCOL)


# In[24]:


with open("bag.pkl", "wb") as f:
    pickle.dump(clfs2, f, pickle.HIGHEST_PROTOCOL)


# # Max Features

# In[25]:


fea_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators = 30,
    bootstrap = True, bootstrap_features = False,
    max_samples = 0.5, max_features = 2)

fea = [fea_clf]


# ## fit the classifier

# In[26]:


fea_clf.fit(X_all_train, y_all_train)


# ## get predictions

# In[27]:


y_pred_train_fea = fea_clf.predict(X_all_train)
y_pred_test_fea = fea_clf.predict(X_all_test)


# ## get accuracies

# In[28]:


acc_train_fea = accuracy_score(y_all_train, y_pred_train_fea)
acc_test_fea = accuracy_score(y_all_test, y_pred_test_fea)
acc_train_fea


# ## pickle the results

# In[29]:


fea_acc_list = [acc_train_fea, acc_test_fea]
fea_acc_list


# In[30]:


with open("acc_fea.pkl", "wb") as f:
    pickle.dump(fea_acc_list, f, pickle.HIGHEST_PROTOCOL)


# In[31]:


with open("fea.pkl", "wb") as f:
    pickle.dump(fea, f, pickle.HIGHEST_PROTOCOL)


# # DataFrame

# In[32]:


fea_accuracies = []
df_fea = pd.DataFrame(columns=["acc_train", "acc_test", "features"])
df_fea


# In[33]:


names = []

for ft in fea_clf.estimators_features_:
    names.append([X_all_train.columns[ft[0]], X_all_train.columns[ft[1]]])


# In[34]:


for i in range(len(fea_clf.estimators_)):
    
    X = dbc["data"][names[i]]
    y = dbc["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    
    clf.fit(X_train, y_train)
    
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    df_fea.loc[len(df_fea.index)] = [acc_train, acc_test, names[i]]


# ## sort dataframe

# In[35]:


df_sorted = df_fea.sort_values(by=["acc_test", "acc_train"], ascending=False)


# ## pickle results

# In[36]:


with open("acc_fea_rank.pkl", "wb") as f:
    pickle.dump(df_sorted, f, pickle.HIGHEST_PROTOCOL)

