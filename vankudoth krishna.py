#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import sklearn


# In[4]:


from sklearn.datasets import load_boston
df=load_boston()


# In[5]:


df.keys()


# In[6]:


print(df.DESCR)


# In[7]:


type(df)


# In[8]:


print(df.feature_names)


# In[9]:


print(df.data)


# In[10]:


print(df.target)


# In[11]:


print(df.filename)


# In[ ]:


#we convert our dataset in to the pandas dataframe,so that is easier to 
Analysis data


# In[12]:


boston=pd.DataFrame(df.data,columns=df.feature_names)
boston.head()


# In[13]:


boston['MEDV']=df.target
boston.head()


# In[ ]:


##check if the data contains any null value or not


# In[14]:


boston.isnull()


# In[15]:


boston.isnull().sum()


# In[17]:


from sklearn.model_selection import train_test_split
X=boston.drop('MEDV',axis=1)
Y=boston['MEDV']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


@@NOW LETS IMPORT THE LINEAR REGREESION MODEL FROM SKLEARN AND TRAIN IT ON 
THE TRAINING DATASET


# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[19]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=5)


# In[21]:


lin_model=LinearRegression()
lin_model.fit(X_train,Y_train)


# In[22]:


y_train_predict=lin_model.predict(X_train)
rmse=(np.sqrt(mean_squared_error(Y_train,y_train_predict)))


# In[24]:


print("the model performance for testing set")
print('RMSE is {}'.format(rmse))
print("\n")


# In[ ]:


#on testing set


# In[25]:


y_test_predict=lin_model.predict(X_test)
rmse=(np.sqrt(mean_squared_error(Y_test,y_test_predict)))


# In[26]:


print("the model performance for testing set")
print('RMSE is {}'.format(rmse))


# In[ ]:




