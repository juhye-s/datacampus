#!/usr/bin/env python
# coding: utf-8

# ## 비행기 탑승 승객 예측
# - 데이터 : sm.datasets.get_rdataset("AirPassengers")

# In[1]:


import pandas as pd
from statsmodels import datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[2]:


raw_set= sm.datasets.get_rdataset("AirPassengers")
raw_set.data


# In[3]:


raw=raw_set.data


# In[4]:


raw.time=pd.date_range('1949-01-01', periods=len(raw), freq='M')
raw


# In[5]:


raw['month']=raw.time.dt.month
raw


# In[6]:


plt.plot(raw.time, raw.value)
plt.show()


# In[7]:


sm.graphics.tsa.plot_acf(raw.value, lags=50, use_vlines=True, title='ACF')
plt.show()


# In[8]:


result=sm.OLS.from_formula(formula='value ~ C(month)-1', data=raw).fit()
result.summary2()


# In[9]:


plt.plot(raw.time, raw.value, raw.time, result.fittedvalues)
plt.show()


# In[10]:


plt.plot(raw.time, result.resid)
plt.show()


# In[11]:


sm.graphics.tsa.plot_acf(result.resid, lags=50, use_vlines=True)
plt.show()


# In[12]:


raw.value.diff(12)[:13]


# In[13]:


plt.plot(raw.time[12:], raw.value.diff(12).dropna())
plt.show()


# In[14]:


sm.graphics.tsa.plot_acf(raw.value.diff(12).dropna(), lags=50, use_vlines=True)
plt.show()


# ### ARMA

# In[15]:


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# In[16]:


import statsmodels
import numpy as np


# In[17]:


raw.head()


# In[18]:


plt.figure(figsize=(10,8))
sm.graphics.tsa.plot_acf(raw.value, lags=30, ax=plt.subplot(211), title='ACF')
sm.graphics.tsa.plot_pacf(raw.value, lags=30, ax=plt.subplot(212), title='PACF')
plt.ylim(-1.1, 1.1)
plt.show()


# In[19]:


raw["logvalue"]=np.log(raw.value)
raw.plot(x='time', y='logvalue')
plt.show()


# In[20]:


plt.figure(figsize=(10,8))
sm.graphics.tsa.plot_acf(raw.logvalue, lags=30, ax=plt.subplot(211), title='log_ACF')
sm.graphics.tsa.plot_pacf(raw.logvalue, lags=30, ax=plt.subplot(212), title='log_PACF')
plt.show()


# In[21]:


fit=statsmodels.tsa.arima_model.ARMA(raw.value,(1,1)).fit()
display(fit.summary())


# In[22]:


fit_log=statsmodels.tsa.arima_model.ARMA(raw.logvalue, (1,1)).fit()
display(fit_log.summary())


# In[23]:


from itertools import product

result=[]
for p,q in product(range(4), range(2)):
    model=statsmodels.tsa.arima_model.ARMA(raw.value, (p,q)).fit()
    result.append({"p":p, "q":q, "LLF":model.llf,"AIC":model.aic,"BIC":model.bic})
    
result=pd.DataFrame(result)


# In[24]:


display(result)


# In[25]:


fit=statsmodels.tsa.arima_model.ARMA(raw.value, (2,1)).fit()
display(fit.summary())


# In[26]:


result_log=[]
for p,q in product(range(4), range(2)):
    model=statsmodels.tsa.arima_model.ARMA(raw.value, (p,q)).fit()
    result_log.append({"p":p, "q":q, "LLF":model.llf,"AIC":model.aic,"BIC":model.bic})
    
result=pd.DataFrame(result_log)


# In[27]:


fit_log=statsmodels.tsa.arima_model.ARMA(raw.value, (2,1)).fit()
display(fit_log.summary())


# ### ARIMA

# In[28]:


import pandas as pd
from statsmodels import datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[29]:


plt.figure(figsize=(10,8))
sm.graphics.tsa.plot_acf(raw.value, lags=35, ax=plt.subplot(211), title='ACF')
sm.graphics.tsa.plot_pacf(raw.value, lags=35, ax=plt.subplot(212), title='PACF')
plt.ylim(-1.1, 1.1)
plt.show()


# In[30]:


fit=sm.tsa.arima.ARIMA(raw.value, order=(1,1,0)).fit()
display(fit.summary())


# In[31]:


fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.show()


# In[32]:


from itertools import product

result=[]
for p,d, q in product(range(4), range(2), range(4)):
    model=sm.tsa.arima.ARIMA(raw.value, order=(p,d,q)).fit()
    result.append({"p":p, "d":d, "q":q, "LLF":model.llf,"AIC":model.aic,"BIC":model.bic})


# In[33]:


result=pd.DataFrame(result)
display(result)


# In[34]:


fit.forecast(steps=3)


# In[ ]:




