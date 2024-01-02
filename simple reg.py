#!/usr/bin/env python
# coding: utf-8

# ## 단순선형회귀분석

# - #### CRIM 변수로 CRIM 이 PRICE 에 영향을 주는지 확인
#     - Boston 주택 가격데이터 이용 

# In[3]:


import pandas as pd
import numpy as np
from sklearn import datasets
#데이터 불러오기
boston= datasets.load_boston()
print(boston.DESCR)


# In[5]:


boston_df=pd.DataFrame(boston.data, columns= boston.feature_names)
boston_df.head()


# In[7]:


boston.data


# In[6]:


boston.feature_names


# In[8]:


boston_df["PRICE"]=pd.DataFrame(boston.target)
boston_df.head()


# In[9]:


boston_df.info()


# In[10]:


boston_df.describe()


# In[11]:


boston_df.shape


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


crim_price= boston_df[["CRIM","PRICE"]]
crim_price.head()


# In[15]:


sns.regplot(x="CRIM",y="PRICE",data= crim_price)
plt.show()


# - #### price(target) ~ crim 선형회귀분석

# In[16]:


price=boston_df[["PRICE"]]
crim=boston_df[["CRIM"]]


# In[18]:


import statsmodels.api as sm


# In[19]:


crim1=sm.add_constant(crim, has_constant='add')


# In[34]:


reg=sm.OLS(price, crim1) #잔차 제곱의 합을 최소화하도록하는 선형회기 추정방식
fitted_model=reg.fit()


# In[25]:


fitted_model.summary()


# - #### y_hat=beta0 + beta1 * X (회귀식) 계산

# In[26]:


fitted_model.params


# In[27]:


np.dot(crim1, fitted_model.params)


# In[28]:


len(np.dot(crim1, fitted_model.params))


# In[30]:


pred=fitted_model.predict(crim1)


# In[32]:


pred


# In[36]:


pred-np.dot(crim1,fitted_model.params)


# - #### 시각화

# In[37]:


import matplotlib.pyplot as plt
plt.scatter(crim, price, label="data")
plt.plot(crim, pred, label="result")
plt.legend()
plt.show()


# In[38]:


fitted_model.resid.plot()
plt.xlabel("residual_number")
plt.show()


# In[40]:


sum(fitted_model.resid)

