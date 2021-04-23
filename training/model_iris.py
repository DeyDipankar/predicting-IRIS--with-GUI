#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl


# In[6]:


df= pd.read_csv('Iris.csv', usecols= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
df.head()


# In[20]:


print(df.shape)


# In[16]:


X= df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[17]:


le = LabelEncoder()
y = le.fit_transform(y)
y= pd.Series(y)
y.value_counts()


# In[19]:


rf = RandomForestClassifier().fit(X,y)


# In[34]:


with open('model.pkl', 'wb') as f:
    pkl.dump(rf, f)


# In[35]:


with open('model.pkl', 'rb') as f:
    model = pkl.load(f)


# In[37]:


#le.inverse_transform(model.predict([[2.5,1.5,1.5,2]]))


# In[42]:


# load the model from disk
loaded_model = pkl.load(open('model.pkl', 'rb'))
result = loaded_model.predict([[2.5,1.5,1.5,2]])
print(result)

