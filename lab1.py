#!/usr/bin/env python
# coding: utf-8

# In[44]:


import os
import tarfile
import urllib
import gzip
import requests
import pandas as pd


# In[6]:


url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"


# In[7]:


urllib.request.urlretrieve(url,'plik1.tgz')


# In[8]:


plik = tarfile.open('plik1.tgz')


# In[9]:


plik.extractall('./')


# In[10]:


plik.close()


# In[35]:


import shutil,sys
filename_in = "housing.csv"
filename_out = "housing.csv.tgz"
with open(filename_in, 'rb') as f_in:
    with gzip.open(filename_out, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[36]:


df = pd.read_csv('housing.csv')


# In[37]:


df


# In[38]:


df.head()


# In[39]:


df.info()


# In[40]:


df['ocean_proximity'].value_counts()


# In[41]:


df['ocean_proximity'].describe()


# In[42]:


import matplotlib.pyplot as plt


# In[19]:


df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')


# In[20]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')


# In[21]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')


# In[23]:


#korelacja = df.corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"index": "atrybuty", "median_house_value": "wspolczynnik_korelacji"})
df.corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={"index": "atrybuty", "median_house_value": "wspolczynnik_korelacji"}).to_csv('korelacja.csv')


# In[24]:


#korelacja.to_csv(index=False)


# In[25]:


pip install sklearn


# In[2]:


pip install seaborn


# In[26]:


import seaborn as sns
sns.pairplot(df)


# In[28]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
len(train_set),len(test_set)


# In[30]:


train_set.corr()['median_house_value'].sort_values(ascending=False)


# In[33]:


train_set.to_pickle('train_set.pkl')


# In[31]:


test_set.corr()['median_house_value'].sort_values(ascending=False)


# In[43]:


test_set.to_pickle('test_set.pkl')


# In[ ]:




