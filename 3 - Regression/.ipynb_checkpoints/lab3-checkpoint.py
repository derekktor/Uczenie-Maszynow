#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


import matplotlib.pyplot as plt


# In[34]:


from sklearn.metrics import mean_squared_error


# In[3]:


size = 300


# ### Target features

# In[4]:


X = np.random.rand(size) * 5 - 2.5


# ### Prediction target

# $$
# y = 1 \cdot x^{4} + 1 \cdot x^{3} + 2 \cdot x^{3} + 1 \cdot x^{2} + -4 \cdot x + 2
# $$

# In[5]:


coefs = [2, -4, 1, 2, 1]
y = coefs[4]*(X**4) + coefs[3]*(X**3) + coefs[2]*(X**2) + coefs[1]*X + coefs[0] + np.random.rand(size)*8-4


# ### Load data

# In[6]:


df = pd.DataFrame({'X': X, 'y': y})


# ### Choose target features

# In[16]:


X = df[['X']]
X.head()


# ### Choose prediction target

# In[17]:


y = df[['y']]
y.head()


# ### Split the data

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# In[19]:


df.to_csv("dane.csv", index=None)


# In[20]:


df.plot.scatter(x='X', y='y')


# ## Linear regressor

# In[21]:


from sklearn.linear_model import LinearRegression


# In[24]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[25]:


sep = "\t\t:\t"


# In[120]:


print(f"""
intercept{sep}{lin_reg.intercept_}
coef{sep}{lin_reg.coef_}
f(x=2){sep}{lin_reg.predict([[2]])}
MSE{sep}{mean_squared_error(y_test, lin_reg.predict(X_test))}
""")


# In[125]:


lin_reg_mse_test = mean_squared_error(y_test, lin_reg.predict(X_test))
lin_reg_mse_train = mean_squared_error(y_train, lin_reg.predict(X_train))
# print(lin_reg_mse_test, lin_reg_mse_train)


# ## K-Nearest Neighbors

# In[37]:


from sklearn.neighbors import KNeighborsRegressor


# #### k = 3

# In[38]:


knn_3_reg = KNeighborsRegressor(n_neighbors=3)


# In[39]:


knn_3_reg.fit(X_train, y_train)


# In[40]:


print(f"""
f(x=2){sep}{knn_3_reg.predict([[2]])}
MSE{sep}{mean_squared_error(y_test, knn_3_reg.predict(X_test))}
""")


# In[126]:


knn_3_reg_mse_test = mean_squared_error(y_test, knn_3_reg.predict(X_test))
knn_3_reg_mse_train = mean_squared_error(y_train, knn_3_reg.predict(X_train))
print(knn_3_reg_mse_test, knn_3_reg_mse_train)


# #### k = 5

# In[41]:


knn_5_reg = KNeighborsRegressor(n_neighbors=5)


# In[98]:


knn_5_reg.fit(X_train, y_train)


# In[99]:


print(f"""
f(x=2){sep}{knn_5_reg.predict([[2]])}
MSE{sep}{mean_squared_error(y_test, knn_5_reg.predict(X_test))}
""")


# In[127]:


knn_5_reg_mse_test = mean_squared_error(y_test, knn_5_reg.predict(X_test))
knn_5_reg_mse_train = mean_squared_error(y_train, knn_5_reg.predict(X_train))
print(knn_5_reg_mse_test, knn_5_reg_mse_train)


# ## Polynomial Regressor

# In[44]:


from sklearn.preprocessing import PolynomialFeatures


# ##### Polynomial Features

# In[115]:


poly_2 = PolynomialFeatures(degree=2, include_bias=False)
poly_3 = PolynomialFeatures(degree=3, include_bias=False)
poly_4 = PolynomialFeatures(degree=4, include_bias=False)
poly_5 = PolynomialFeatures(degree=5, include_bias=False)


# ##### Transform X

# In[116]:


X_train_2 = poly_2.fit_transform(X_train)
X_test_2 = poly_2.fit_transform(X_test)

X_train_3 = poly_3.fit_transform(X_train)
X_test_3 = poly_3.fit_transform(X_test)

X_train_4 = poly_4.fit_transform(X_train)
X_test_4 = poly_4.fit_transform(X_test)

X_train_5 = poly_5.fit_transform(X_train)
X_test_5 = poly_5.fit_transform(X_test)


# In[117]:


lin_2_reg = LinearRegression()
lin_2_reg.fit(X_train_2, y_train)

lin_3_reg = LinearRegression()
lin_3_reg.fit(X_train_3, y_train)

lin_4_reg = LinearRegression()
lin_4_reg.fit(X_train_4, y_train)

lin_5_reg = LinearRegression()
lin_5_reg.fit(X_train_5, y_train)


# In[118]:


# print(f"y-intercept: {lin_2_reg.intercept_[0]}")
# ind = 0
# for c in lin_2_reg.coef_[0]:
#     print(f"coef{ind}: {c}")
#     ind += 1


# $$
#     y = 6.5 \cdot x^{2} + 3.5 \cdot x -1.3
# $$

# In[119]:


# print(lin_2_reg.coef_[0][1] * 2**2 + lin_2_reg.coef_[0][0] * 2 + lin_reg.intercept_[0])


# In[114]:


pred_2 = lin_2_reg.predict(X_test_2)
print(f"MSE: {mean_squared_error(y_test, pred_2)}")


# In[128]:


poly_2_reg_mse_test = mean_squared_error(y_test, lin_2_reg.predict(X_test_2))
poly_2_reg_mse_train = mean_squared_error(y_train, lin_2_reg.predict(X_train_2))
print(poly_2_reg_mse_test, poly_2_reg_mse_train)


# In[129]:


poly_3_reg_mse_test = mean_squared_error(y_test, lin_3_reg.predict(X_test_3))
poly_3_reg_mse_train = mean_squared_error(y_train, lin_3_reg.predict(X_train_3))
print(poly_3_reg_mse_test, poly_3_reg_mse_train)


# In[130]:


poly_4_reg_mse_test = mean_squared_error(y_test, lin_4_reg.predict(X_test_4))
poly_4_reg_mse_train = mean_squared_error(y_train, lin_4_reg.predict(X_train_4))
print(poly_4_reg_mse_test, poly_4_reg_mse_train)


# In[131]:


poly_5_reg_mse_test = mean_squared_error(y_test, lin_5_reg.predict(X_test_5))
poly_5_reg_mse_train = mean_squared_error(y_train, lin_5_reg.predict(X_train_5))
print(poly_5_reg_mse_test, poly_5_reg_mse_train)


# #### DataFrame containing MSE values

# In[135]:


train_mse = [lin_reg_mse_train, knn_3_reg_mse_train, knn_5_reg_mse_train, poly_2_reg_mse_train, poly_3_reg_mse_train, poly_4_reg_mse_train, poly_5_reg_mse_train]
test_mse = [lin_reg_mse_test, knn_3_reg_mse_test, knn_5_reg_mse_test, poly_2_reg_mse_test, poly_3_reg_mse_test, poly_4_reg_mse_test, poly_5_reg_mse_test]

index_mse = ["lin_reg", "knn_3_reg", "knn_5_reg", "poly_2_reg", "poly_3_reg", "poly_4_reg", "poly_5_reg"]


# In[136]:


df_mse = pd.DataFrame({"train_mse": train_mse, "test_mse": test_mse}, index=index_mse)
df_mse


# In[137]:


import pickle


# In[138]:


with open('mse.pkl', 'wb') as f:
    pickle.dump(df_mse, f, pickle.HIGHEST_PROTOCOL)


# In[141]:


list_reg = [(lin_reg, None), (knn_3_reg, None), (knn_5_reg, None), (lin_2_reg, poly_2), (lin_3_reg, poly_3), (lin_4_reg, poly_4), (lin_5_reg, poly_5)]
with open('reg.pkl', 'wb') as f:
    pickle.dump(list_reg, f, pickle.HIGHEST_PROTOCOL)


# In[142]:


with open('mse.pkl', 'rb') as f:
    print(pickle.load(f))
    
with open('reg.pkl', 'rb') as f:
    print(pickle.load(f))


# In[ ]:




