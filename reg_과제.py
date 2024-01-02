#!/usr/bin/env python
# coding: utf-8

# ### 1. 단순선형회귀 
# #### RM변수와 LSTAT변수로 price를 예측하도록 각각 단순선형회귀분석 후 결과 분석
# 
# - Boston 주택 가격데이터 이용

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
boston= datasets.load_boston()
print(boston.DESCR)


# In[5]:


boston_df=pd.DataFrame(boston.data, columns= boston.feature_names)
boston_df.head()


# In[6]:


boston_df["PRICE"]=pd.DataFrame(boston.target)
boston_df.head()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


#RM 변수로 단순 선형 회귀 분석
rm_price= boston_df[["RM","PRICE"]]
sns.regplot(x="RM",y="PRICE",data= rm_price)
plt.show()


# In[18]:


price=boston_df[["PRICE"]]
rm=boston_df[["RM"]]


# In[19]:


import statsmodels.api as sm
rm1=sm.add_constant(rm, has_constant='add')
reg=sm.OLS(price, rm1) #잔차 제곱의 합을 최소화하도록하는 선형회기 추정방식
fitted_model=reg.fit()
pred=fitted_model.predict(rm1)


# In[21]:


import matplotlib.pyplot as plt
plt.scatter(rm, price, label="data")
plt.plot(rm, pred, label="result")
plt.legend()
plt.show()


# In[22]:


fitted_model.resid.plot()
plt.xlabel("residual_number")
plt.show()


# In[23]:


#LSTAT 변수로 단순 선형 회귀 분석
lstat_price= boston_df[["LSTAT","PRICE"]]
sns.regplot(x="LSTAT",y="PRICE",data= lstat_price)
plt.show()


# In[25]:


price=boston_df[["PRICE"]]
lstat=boston_df[["LSTAT"]]
import statsmodels.api as sm
lstat1=sm.add_constant(lstat, has_constant='add')
reg=sm.OLS(price, lstat1) 
fitted_model=reg.fit()
pred2=fitted_model.predict(lstat1)
import matplotlib.pyplot as plt
plt.scatter(lstat, price, label="data")
plt.plot(lstat, pred2, label="result")
plt.legend()
plt.show()


# In[26]:


fitted_model.resid.plot()
plt.xlabel("residual_number")
plt.show()


# ###  2. 다중선형회귀와 단순선형회귀계수 비교
# #### CRIM, RM, LSTAT 세개 변수로 다중선형회귀 적합한 결과와  각각의 변수를 단순선형회귀 적합한 모델의 회귀계수를 비교 
# 
# - Boston 주택 가격데이터 이용

# In[28]:


import pandas as pd
import numpy as np
from sklearn import datasets
import statsmodels.api as sm
boston= datasets.load_boston()
boston_df=pd.DataFrame(boston.data, columns= boston.feature_names)
boston_df.head()


# In[29]:


boston_df["PRICE"]=pd.DataFrame(boston.target)
boston_df.head()
x_data=boston_df[['CRIM','RM','LSTAT']]
x_data.head()


# In[30]:


price=boston_df[["PRICE"]]
x_data1=sm.add_constant(x_data, has_constant='add')
multi_model=sm.OLS(price, x_data1)
fitted_multi_model=multi_model.fit()
fitted_multi_model.summary()


# In[31]:


#각각 단순선형회귀 비교
price=boston_df[["PRICE"]]
crim=boston_df[["CRIM"]]
crim1=sm.add_constant(crim, has_constant='add')
rm=boston_df[["RM"]]
rm1=sm.add_constant(rm, has_constant='add')
lstat=boston_df[["LSTAT"]]
lstat1=sm.add_constant(lstat, has_constant='add')


# In[34]:


model1=sm.OLS(price,crim1)
fitted_model1=model1.fit()

model2=sm.OLS(price,rm1)
fitted_model2=model2.fit()

model3=sm.OLS(price,lstat1)
fitted_model3=model3.fit()


# In[33]:


fitted_model1.resid.plot(label="crim")
fitted_multi_model.resid.plot(label="full")
plt.legend()


# In[35]:


fitted_model2.resid.plot(label="rm")
fitted_multi_model.resid.plot(label="full")
plt.legend()


# In[36]:


fitted_model3.resid.plot(label="lstat")
fitted_multi_model.resid.plot(label="full")
plt.legend()


# ### 3. 다중공선성과 회귀모델 성능 확인
# ####  (1) CRIM, RM, LSTAT, B, TAX, AGE, ZN, NOX, INDUS 변수로 데이터를 분할 하여 회귀 모형 생성 후 성능확인
# #### (2) 다중공선성을 확인하여 변수 제거 후 모형의 성능 높이기
# 
# - Boston 주택 가격데이터 이용
# 

# In[38]:


from sklearn.model_selection import train_test_split
X=x_data1
y=price
x_train, x_test, y_train, y_test= train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=102)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[44]:


x_train1=sm.add_constant(x_train, has_constant='add')
fit_1=sm.OLS(y_train, x_train1)
fit_1=fit_1.fit()


# In[45]:


x1_test1=sm.add_constant(x_test, has_constant='add')


# In[46]:


plt.plot(np.array(fit_1.predict(x1_test1)),label="pred2")
plt.plot(np.array(y_test),label="true")
plt.legend()
plt.show()


# In[ ]:




