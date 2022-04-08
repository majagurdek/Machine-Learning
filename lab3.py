#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y}) 
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


#Linear Regression


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


X_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# In[11]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.intercept_)


# In[12]:


from sklearn.metrics import mean_squared_error


# In[13]:


y_pred1 = lin_reg.predict(X_train)


# In[14]:


train_mse_linear = mean_squared_error(y_train, y_pred1)


# In[15]:


train_mse_linear


# In[16]:


y_pred2 = lin_reg.predict(X_test)


# In[17]:


test_mse_linear = mean_squared_error(y_test, y_pred2)


# In[18]:


test_mse_linear


# In[45]:


results = pd.DataFrame(columns=['train_mse', 'test_mse'])


# In[46]:


results.loc['lin_reg']=[train_mse_linear,test_mse_linear]


# In[47]:


results


# In[48]:


#KNN 3


# In[49]:


import sklearn.neighbors


# In[50]:


knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)


# In[51]:


knn_3_reg.fit(X_train, y_train.ravel())


# In[52]:


y_pred_knn3 = knn_3_reg.predict(X_train)


# In[53]:


train_mse_knn3 = mean_squared_error(y_train, y_pred_knn3)


# In[54]:


train_mse_knn3


# In[55]:


y_pred_knn3_2 = knn_3_reg.predict(X_test)


# In[56]:


test_mse_knn3 = mean_squared_error(y_test, y_pred_knn3_2)


# In[57]:


test_mse_knn3


# In[73]:


results.loc['knn_3_reg']=[train_mse_knn3,test_mse_knn3]


# In[59]:


results


# In[60]:


#KNN 5


# In[61]:


knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)


# In[62]:


knn_5_reg.fit(X_train, y_train.ravel())


# In[63]:


y_pred_knn5 = knn_5_reg.predict(X_train)


# In[64]:


train_mse_knn5 = mean_squared_error(y_train, y_pred_knn5)


# In[65]:


train_mse_knn5


# In[66]:


y_pred_knn5_2 = knn_5_reg.predict(X_test)


# In[67]:


test_mse_knn5 = mean_squared_error(y_test, y_pred_knn5_2)


# In[68]:


test_mse_knn5


# In[74]:


results.loc['knn_5_reg']=[train_mse_knn5,test_mse_knn5]


# In[75]:


results


# In[76]:


#Wielomianowa 2


# In[85]:


from sklearn.preprocessing import PolynomialFeatures


# In[86]:


poly_features2 = PolynomialFeatures(degree=2, include_bias=False)


# In[87]:


X_poly2 = poly_features2.fit_transform(X_train)


# In[88]:


poly_reg2 = LinearRegression()


# In[89]:


poly_reg2.fit(X_poly2, y_train)


# In[92]:


y_pred_train2 = poly_reg2.predict(poly_features2.fit_transform(X_train))


# In[94]:


train_mse_p2 = mean_squared_error(y_train, y_pred_train2)


# In[95]:


train_mse_p2


# In[104]:


y_pred_test2 = poly_reg2.predict(poly_features2.fit_transform(X_test))


# In[105]:


test_mse_p2 = mean_squared_error(y_test, y_pred_test2)


# In[106]:


test_mse_p2


# In[107]:


results.loc['poly_2_reg']=[train_mse_p2,test_mse_p2]


# In[108]:


results


# In[ ]:


#Wielomianowa 3


# In[110]:


poly_features3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly_features3.fit_transform(X_train)
poly_reg3 = LinearRegression()
poly_reg3.fit(X_poly3, y_train)


# In[113]:


y_pred_train3 = poly_reg3.predict(poly_features3.fit_transform(X_train))
train_mse_p3 = mean_squared_error(y_train, y_pred_train3)
train_mse_p3


# In[115]:


y_pred_test3 = poly_reg3.predict(poly_features3.fit_transform(X_test))
test_mse_p3 = mean_squared_error(y_test, y_pred_test3)
test_mse_p3


# In[116]:


results.loc['poly_3_reg']=[train_mse_p3,test_mse_p3]


# In[117]:


results


# In[ ]:


#Wielomianowa 4


# In[118]:


poly_features4 = PolynomialFeatures(degree=4, include_bias=False)
X_poly4 = poly_features4.fit_transform(X_train)
poly_reg4 = LinearRegression()
poly_reg4.fit(X_poly4, y_train)


# In[119]:


y_pred_train4 = poly_reg4.predict(poly_features4.fit_transform(X_train))
train_mse_p4 = mean_squared_error(y_train, y_pred_train4)
train_mse_p4


# In[120]:


y_pred_test4 = poly_reg4.predict(poly_features4.fit_transform(X_test))
test_mse_p4 = mean_squared_error(y_test, y_pred_test4)
test_mse_p4


# In[121]:


results.loc['poly_4_reg']=[train_mse_p4,test_mse_p4]


# In[122]:


results


# In[ ]:


#Wielomianowa 5


# In[123]:


poly_features5 = PolynomialFeatures(degree=5, include_bias=False)
X_poly5 = poly_features5.fit_transform(X_train)
poly_reg5 = LinearRegression()
poly_reg5.fit(X_poly5, y_train)


# In[125]:


y_pred_train5 = poly_reg5.predict(poly_features5.fit_transform(X_train))
train_mse_p5 = mean_squared_error(y_train, y_pred_train5)
train_mse_p5


# In[126]:


y_pred_test5 = poly_reg5.predict(poly_features5.fit_transform(X_test))
test_mse_p5 = mean_squared_error(y_test, y_pred_test5)
test_mse_p5


# In[127]:


results.loc['poly_5_reg']=[train_mse_p5,test_mse_p5]


# In[128]:


results


# In[131]:


import pickle


# In[132]:


with open('mse.pkl', 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


# In[133]:


with open('mse.pkl', 'rb') as f:
    print(pickle.load(f))


# In[134]:


reg = [(lin_reg, None),(knn_3_reg, None),(knn_5_reg, None), (poly_reg2,poly_features2), (poly_reg3,poly_features3), (poly_reg4,poly_features4), (poly_reg5,poly_features5)]


# In[135]:


reg


# In[136]:


with open('reg.pkl', 'wb') as f:
    pickle.dump(reg, f, pickle.HIGHEST_PROTOCOL)


# In[139]:


with open('reg.pkl', 'rb') as f:
    print(pickle.load(f))

