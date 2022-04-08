#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[6]:


from sklearn.datasets import fetch_openml 
mnist = fetch_openml('mnist_784', version=1)


# In[7]:


print((np.array(mnist.data.loc[42]).reshape(28, 28) > 0).astype(int))


# In[8]:


mnist.target


# In[9]:


mnist.data


# In[10]:


y = mnist.target


# In[11]:


X = mnist.data


# In[12]:


y


# In[13]:


y = y.sort_values()


# In[14]:


y


# In[15]:


X


# In[16]:


X = X.reindex(y.index)


# In[17]:


X


# In[18]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[19]:


y_train.unique()


# In[20]:


y_test.unique()


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


y_train.unique()


# In[24]:


y_test.unique()


# In[25]:


y_train_0 = (y_train == '0')


# In[26]:


y_test_0 = (y_test == '0')


# In[27]:


print(y_train_0)
print(np.unique(y_train_0))
print(len(y_train_0))


# In[28]:


from sklearn.linear_model import SGDClassifier


# In[29]:


sgd_clf = SGDClassifier(random_state=42)


# In[30]:


sgd_clf.fit(X_train, y_train_0)


# In[31]:


from sklearn.model_selection import cross_val_score


# In[40]:


from sklearn.model_selection import cross_val_predict


# In[41]:


from sklearn.metrics import confusion_matrix


# In[42]:


import pickle


# In[43]:


acc = [sgd_clf.score(X_train, y_train_0), sgd_clf.score(X_test, y_test_0)]
with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(acc, f, pickle.HIGHEST_PROTOCOL)


# In[44]:


acc


# In[45]:


score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)
with open('sgd_cva.pkl', 'wb') as f:
    pickle.dump(score, f, pickle.HIGHEST_PROTOCOL)


# In[46]:


score


# In[47]:


# wiele klas 
sgd_m_clf = SGDClassifier(random_state=42,n_jobs=-1)


# In[48]:


sgd_m_clf.fit(X_train, y_train)


# In[50]:


y_train_pred = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)


# In[54]:


conf_mx = confusion_matrix(y_train, y_train_pred)
with open('sgd_cmx.pkl', 'wb') as f:
    pickle.dump(conf_mx, f, pickle.HIGHEST_PROTOCOL)


# In[55]:


conf_mx


# In[ ]:




