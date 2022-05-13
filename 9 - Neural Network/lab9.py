#!/usr/bin/env python
# coding: utf-8

# # imports

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


from sklearn.datasets import load_iris


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.linear_model import Perceptron


# In[7]:


import seaborn as sns


# In[1]:


import pickle


# In[2]:


import tensorflow as tf
from tensorflow import keras


# # load data

# In[8]:


iris = load_iris(as_frame=True)


# In[43]:


X_ir = iris.data[['petal length (cm)', 'petal width (cm)']]


# In[44]:


y_ir = iris.target


# In[45]:


y0 = (y_ir == 0).astype(int)
y1 = (y_ir == 1).astype(int)
y2 = (y_ir == 2).astype(int)


# In[46]:


X0_train, X0_test, y0_train, y0_test = train_test_split(X_ir, y0, test_size=.2)
X1_train, X1_test, y1_train, y1_test = train_test_split(X_ir, y1, test_size=.2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X_ir, y2, test_size=.2)


# # visualization

# In[47]:


def draw(X, y):
    pd.concat([X, y], axis=1).plot.scatter(
        x='petal length (cm)',
        y='petal width (cm)',
        c='target',
        colormap='viridis'
    )
    
draw(iris.data, iris.target)


# ## Class 0

# In[59]:


pd.concat([X_ir, y0], axis=1).sample(5)


# In[53]:


pd.concat([X_ir, y0], axis=1).plot.scatter(
    x = 'petal length (cm)',
    y = 'petal width (cm)',
    c = 'target',
    colormap = 'viridis'
)


# ## Class 1

# In[57]:


pd.concat([X_ir, y1], axis=1).sample(5)


# In[56]:


pd.concat([X_ir, y1], axis=1).plot.scatter(
    x = 'petal length (cm)',
    y = 'petal width (cm)',
    c = 'target',
    colormap = 'viridis'
)


# ## Class 2

# In[60]:


pd.concat([X_ir, y2], axis=1).sample(5)


# In[61]:


pd.concat([X_ir, y2], axis=1).plot.scatter(
    x = 'petal length (cm)',
    y = 'petal width (cm)',
    c = 'target',
    colormap = 'viridis'
)


# # perceptrons

# In[14]:


per0 = Perceptron()
per1 = Perceptron()
per2 = Perceptron()


# ## fit perceptrons

# In[15]:


per1.fit(X1_train, y1_train)
per0.fit(X0_train, y0_train)
per2.fit(X2_train, y2_train)


# ## get predictions

# In[16]:


# bit useless, when using score() method

y0_pred = pd.Series(per0.predict(X0_test))
y1_pred = pd.Series(per1.predict(X1_test))
y2_pred = pd.Series(per2.predict(X2_test))


# ## get classes

# In[33]:


print(f"""
{per0.classes_}
{per0.coef_}
""")


# ## get accuracy scores from perceptrons

# In[20]:


a0_tr = per0.score(X0_train, y0_train)
a0_te = per0.score(X0_test, y0_test)

a1_tr = per1.score(X1_train, y1_train)
a1_te = per1.score(X1_test, y1_test)

a2_tr = per2.score(X2_train, y2_train)
a2_te = per2.score(X2_test, y2_test)

print(f"""
tr\tte

{a0_tr}\t{a0_te}
{a1_tr}\t{a1_te}
{a2_tr}\t{a2_te}
""")


# In[21]:


per_acc = []
per_acc.append((a0_tr, a0_te))
per_acc.append((a1_tr, a1_te))
per_acc.append((a2_tr, a2_te))


# ### pickle accuracies

# In[71]:


with open("per_acc.pkl", "wb") as f:
    pickle.dump(per_acc, f, pickle.HIGHEST_PROTOCOL)


# ## get weights

# In[81]:


print(f"""
{per0.intercept_}\t{per0.coef_[0][0]}\t{per0.coef_[0][1]}
{per1.intercept_}\t{per1.coef_[0][0]}\t{per1.coef_[0][1]}
{per2.intercept_}\t{per2.coef_[0][0]}\t{per2.coef_[0][1]}
""")


# In[88]:


per_wgt = []
per_wgt.append((per0.intercept_[0], per0.coef_[0][0], per0.coef_[0][1]))
per_wgt.append((per1.intercept_[0], per1.coef_[0][0], per1.coef_[0][1]))
per_wgt.append((per2.intercept_[0], per2.coef_[0][0], per2.coef_[0][1]))


# ### pickle weights

# In[87]:


with open("per_wgt.pkl", "wb") as f:
    pickle.dump(per_wgt, f, pickle.HIGHEST_PROTOCOL)


# # Perceptron XOR

# In[37]:


X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor= np.array([0, 1, 1, 0])


# In[38]:


per_xor = Perceptron()


# In[39]:


per_xor.fit(X, y)


# In[40]:


per_xor.predict(X)


# In[41]:


per_xor.score(X, y)


# In[42]:


per_xor.coef_


# # MLP XOR

# In[244]:


opt = keras.optimizers.Adam(learning_rate=0.1)


# In[255]:


mlp = keras.models.Sequential([
    keras.layers.Dense(2, activation="tanh"),
    keras.layers.Dense(1, activation="sigmoid")
])

mlp.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = opt,
    metrics=["mae"]
)

history = mlp.fit(X_xor, y_xor, epochs=100, verbose=False)

print(f"""
{history.history['loss'][0]}
{history.history['loss'][-1]}
""")


# In[257]:


w = mlp.get_weights()


# In[258]:


with open("mlp_xor_weights.pkl", "wb") as f:
    pickle.dump(w, f, pickle.HIGHEST_PROTOCOL)

