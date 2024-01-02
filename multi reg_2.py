#!/usr/bin/env python
# coding: utf-8

# ### 사이킷런으로 회귀분석

# - from sklearn.linear_model import LinearRegression

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
boston=datasets.load_boston()


# In[2]:


boston_df=pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df["PRICE"]=pd.DataFrame(boston.target)
boston_df.head()


# In[6]:


from sklearn.model_selection import train_test_split
X= boston_df[['CRIM','RM','LSTAT']]
y=boston_df[["PRICE"]]
x_train, x_test, y_train, y_test= train_test_split(X,y,train_size=0.7, test_size=0.3, random_state=102)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr=LinearRegression()
lr.fit(x_train, y_train)


# In[13]:


y_pred=lr.predict(x_test)
y_pred


# In[14]:


mse=mean_squared_error(y_test, y_pred)
mse


# In[15]:


r2=r2_score(y_test,y_pred)
r2


# In[23]:


print("절편(베타0):",lr.intercept_)
print("회귀계수:",lr.coef_)


# In[26]:


coeff=pd.DataFrame(lr.coef_.T, index=X.columns, columns=["coeff"])
coeff


# In[ ]:




