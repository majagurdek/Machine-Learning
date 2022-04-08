#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets


# In[2]:


data_breast_cancer = datasets.load_breast_cancer()
print(data_breast_cancer['DESCR'])


# In[3]:


X = data_breast_cancer["data"][:, (3, 4)] 


# In[4]:


y = data_breast_cancer["target"]


# In[5]:


data_iris = datasets.load_iris()
print(data_iris['DESCR'])


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[8]:


from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[9]:


#bez skalowania


# In[10]:


svm_clf1 = LinearSVC(C=1,loss="hinge")


# In[11]:


svm_clf1.fit(X, y)


# In[12]:


y_pred1 = svm_clf1.predict(X_test)


# In[13]:


from sklearn.metrics import accuracy_score


# In[14]:


acc_1_test = accuracy_score(y_test, y_pred1)


# In[15]:


acc_1_test


# In[16]:


y_pred2 = svm_clf1.predict(X_train)


# In[17]:


acc_1_train = accuracy_score(y_train, y_pred2)


# In[18]:


acc_1_train


# In[19]:


#ze skalowaniem 


# In[20]:


svm_clf2 = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge"))])


# In[21]:


svm_clf2.fit(X, y)


# In[22]:


y_pred3 = svm_clf2.predict(X_test)


# In[23]:


acc_2_test = accuracy_score(y_test, y_pred3)


# In[24]:


acc_2_test


# In[25]:


y_pred4 = svm_clf2.predict(X_train)


# In[26]:


acc_2_train = accuracy_score(y_train, y_pred4)


# In[27]:


acc_2_train


# In[29]:


list1 = [acc_1_train, acc_1_test, acc_2_train, acc_2_test]


# In[30]:


import pickle


# In[32]:


with open('bc_acc.pkl', 'wb') as f:
    pickle.dump(list1, f, pickle.HIGHEST_PROTOCOL)


# In[33]:


with open('bc_acc.pkl', 'rb') as f:
    print(pickle.load(f))


# In[34]:


# Irysy


# In[35]:


X2 = data_iris["data"][:, (2, 3)]


# In[36]:


y2 = (data_iris["target"] ==2).astype(np.float64)


# In[37]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2)


# In[38]:


#irys bez skalowania


# In[39]:


svm_clf_i1 = LinearSVC(C=1,loss="hinge")


# In[40]:


svm_clf_i1.fit(X2, y2)


# In[41]:


y_pred_i1 = svm_clf_i1.predict(X2_test)


# In[42]:


acc_i1_test = accuracy_score(y2_test, y_pred_i1)


# In[43]:


acc_i1_test


# In[44]:


y_pred_i2 = svm_clf_i1.predict(X2_train)


# In[45]:


acc_i1_train = accuracy_score(y2_train, y_pred_i2)


# In[46]:


acc_i1_train


# In[47]:


#irysy ze skalowaniem


# In[48]:


svm_clf_i2 = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge"))])


# In[49]:


svm_clf_i2.fit(X2, y2)


# In[50]:


y_pred_i3 = svm_clf_i2.predict(X2_test)


# In[51]:


acc_i2_test = accuracy_score(y2_test, y_pred_i3)


# In[52]:


acc_i2_test


# In[53]:


y_pred_i4 = svm_clf_i2.predict(X2_train)


# In[54]:


acc_i2_train = accuracy_score(y2_train, y_pred_i4)


# In[55]:


acc_i2_train


# In[56]:


#tutaj dodac wyniki do listy, piklowac


# In[57]:


list2 = [acc_i1_train, acc_i1_test, acc_i2_train, acc_i2_test]


# In[58]:


with open('iris_acc.pkl', 'wb') as f:
    pickle.dump(list2, f, pickle.HIGHEST_PROTOCOL)


# In[59]:


with open('iris_acc.pkl', 'rb') as f:
    print(pickle.load(f))


# In[ ]:




