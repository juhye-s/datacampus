#!/usr/bin/env python
# coding: utf-8

# ## 다중선형회귀분석

# - #### CRIM, RM, LSTAT 세개 변수가  PRICE 에 영향을 주는지 확인
#     - Boston 주택 가격데이터 이용 

# - import statsmodels.api as sm

# In[6]:


import pandas as pd
import numpy as np
from sklearn import datasets
import statsmodels.api as sm
boston= datasets.load_boston()
print(boston.DESCR)


# In[7]:


boston_df=pd.DataFrame(boston.data, columns= boston.feature_names)
boston_df.head()


# In[19]:


boston_df["PRICE"]=pd.DataFrame(boston.target)
boston_df.head()


# In[10]:


x_data=boston_df[['CRIM','RM','LSTAT']]
x_data.head()


# In[13]:


price=boston_df[["PRICE"]]


# In[14]:


x_data1=sm.add_constant(x_data, has_constant='add')


# In[17]:


multi_model=sm.OLS(price, x_data1)
fitted_multi_model=multi_model.fit()
fitted_multi_model.summary()


# - #### 단순선형회귀모델의 회귀계수와 비교

# In[22]:


price=boston_df[["PRICE"]]
crim=boston_df[["CRIM"]]


# In[33]:


crim1=sm.add_constant(crim, has_constant='add')


# In[27]:


model1=sm.OLS(price,crim1)
fitted_model1=model1.fit()
print(fitted_model1.params)


# In[28]:


print(fitted_multi_model.params)


# - #### 시각화

# In[30]:


import matplotlib.pyplot as plt
fitted_multi_model.resid.plot()
plt.xlabel("residual_number")
plt.show()


# In[34]:


fitted_model1.resid.plot(label="crim")
fitted_multi_model.resid.plot(label="full")
plt.legend()


# - #### 상관계수/산점도를 통해 다중공선성 확인

# In[35]:


x_data.corr()


# In[36]:


x_data.head()


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns
 
sns.heatmap(x_data.corr(),annot=True)
plt.show()


# In[40]:


sns.pairplot(x_data)
plt.show()


# - #### VIF를 통한 다중공선성 확인

# - from statsmodels.stats.outliers_influence import variance_inflation_factor

# In[43]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=pd.DataFrame()
vif["VIF Factor"]=[variance_inflation_factor(x_data.values, i) for i in range(x_data.shape[1])]
vif["features"]=x_data.columns
vif


# In[47]:


#LSTAT제거
vif=pd.DataFrame()
x_data2=x_data.drop('LSTAT',axis=1)
vif["VIF Factor"]=[variance_inflation_factor(x_data2.values, i) for i in range(x_data2.shape[1])]
vif["features"]=x_data2.columns
vif


# In[49]:


x_data3=sm.add_constant(x_data2, has_constant='add')
model_vif=sm.OLS(price,x_data3)
fitted_model_vif=model_vif.fit()
fitted_model_vif.summary()


# In[50]:


fitted_multi_model.summary()


# - #### 학습 / 검증데이터 분할

# - from sklearn.model_selection import train_test_split

# In[63]:


from sklearn.model_selection import train_test_split
X=x_data2
y=price
x2_train, x2_test, y2_train, y2_test= train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=102)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[70]:


x2_train2=sm.add_constant(x2_train, has_constant='add')
fit_2=sm.OLS(y2_train, x2_train2)
fit_2=fit_2.fit()


# In[71]:


x2_test2=sm.add_constant(x2_test, has_constant='add')


# In[72]:


plt.plot(np.array(fit_2.predict(x2_test2)),label="pred2")
plt.plot(np.array(y2_test),label="true")
plt.legend()
plt.show()


# - #### MSE를 통한 검증데이터에 대한 성능비교

# - from sklearn.metrics import mean_squared_error

# In[74]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_true=y_test['PRICE'],y_pred=fit_1.predict(x_test2))


# In[75]:


mean_squared_error(y_true=y2_test['PRICE'],y_pred=fit_2.predict(x2_test2))


# In[ ]:


#MSE-> 낮을수록 좋음

