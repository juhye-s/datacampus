#!/usr/bin/env python
# coding: utf-8

# ### ARMA

# In[2]:


import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt


# #### ARMA(1,0) = AR(1)

# In[3]:


np.random.seed(123)
ar_params=np.array([0.75])
ma_params=np.array([])
index_name=['const','ar(1)']
ahead=100
ar, ma=np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order=len(ar)-1, len(ma)-1


# In[5]:


y=statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit=statsmodels.tsa.arima_model.ARMA(y,(ar_order,ma_order)).fit(trend='c', disp=0)


# In[7]:


pd.DataFrame(y).plot(figsize=(12,3))
plt.show()

plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()


# In[9]:


fit.forecast(steps=5)


# In[10]:


pred_ts_point=fit.forecast(steps=ahead)[0]
pred_ts_interval=fit.forecast(steps=ahead)[2]


# In[12]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
forecast_index=[i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15 )
plt.legend(['observed', 'forecast'])


# In[11]:


display(fit.summary2())


# In[ ]:





# #### ARMA(2,0) = AR(2)

# In[19]:


np.random.seed(123)
ar_params=np.array([0.75,-0.25])
ma_params=np.array([])
index_name=['const','ar(1)','ar(2)']
ahead=100
ar, ma=np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order=len(ar)-1, len(ma)-1


# In[20]:


y=statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit=statsmodels.tsa.arima_model.ARMA(y,(ar_order,ma_order)).fit(trend='c', disp=0)


# In[21]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
plt.show()

plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()


# In[23]:


pred_ts_point=fit.forecast(steps=ahead)[0]
pred_ts_interval=fit.forecast(steps=ahead)[2]


# In[24]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
forecast_index=[i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15 )
plt.legend(['observed', 'forecast'])


# In[25]:


display(fit.summary2())


# #### ARMA(0,1) = MA(1)

# In[26]:


np.random.seed(123)
ar_params=np.array([])
ma_params=np.array([0.65])
index_name=['const','ma(1)']
ahead=100
ar, ma=np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order=len(ar)-1, len(ma)-1


# In[27]:


y=statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit=statsmodels.tsa.arima_model.ARMA(y,(ar_order,ma_order)).fit(trend='c', disp=0)


# In[28]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
plt.show()

plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()


# In[29]:


pred_ts_point=fit.forecast(steps=ahead)[0]
pred_ts_interval=fit.forecast(steps=ahead)[2]


# In[30]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
forecast_index=[i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15 )
plt.legend(['observed', 'forecast'])


# In[31]:


display(fit.summary2())


# #### ARMA(0,2) = MA(2)

# In[32]:


np.random.seed(123)
ar_params=np.array([])
ma_params=np.array([0.65,-0.25])
index_name=['const','ma(1)','ma(2)']
ahead=100
ar, ma=np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order=len(ar)-1, len(ma)-1


# In[33]:


y=statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit=statsmodels.tsa.arima_model.ARMA(y,(ar_order,ma_order)).fit(trend='c', disp=0)


# In[34]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
plt.show()

plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()


# In[35]:


pred_ts_point=fit.forecast(steps=ahead)[0]
pred_ts_interval=fit.forecast(steps=ahead)[2]


# In[36]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
forecast_index=[i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15 )
plt.legend(['observed', 'forecast'])


# In[37]:


display(fit.summary2())


# #### ARMA(1,1)

# In[38]:


np.random.seed(123)
ar_params=np.array([0.75])
ma_params=np.array([0.65])
index_name=['const','ar(1)','ma(1)']
ahead=100
ar, ma=np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order=len(ar)-1, len(ma)-1


# In[39]:


y=statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit=statsmodels.tsa.arima_model.ARMA(y,(ar_order,ma_order)).fit(trend='c', disp=0)


# In[40]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
plt.show()

plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()


# In[41]:


pred_ts_point=fit.forecast(steps=ahead)[0]
pred_ts_interval=fit.forecast(steps=ahead)[2]


# In[42]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
forecast_index=[i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15 )
plt.legend(['observed', 'forecast'])


# In[43]:


display(fit.summary2())


# #### ARMA(2,2)

# In[44]:


np.random.seed(123)
ar_params=np.array([0.75, -0.25])
ma_params=np.array([0.65,0.5])
index_name=['const','ar(1)','ar(2)','ma(1)','ma(2)']
ahead=100
ar, ma=np.r_[1, -ar_params], np.r_[1, ma_params]
ar_order, ma_order=len(ar)-1, len(ma)-1


# In[45]:


y=statsmodels.tsa.arima_process.arma_generate_sample(ar, ma, nsample=1000, burnin=500)
fit=statsmodels.tsa.arima_model.ARMA(y,(ar_order,ma_order)).fit(trend='c', disp=0)


# In[46]:


pred_ts_point=fit.forecast(steps=ahead)[0]
pred_ts_interval=fit.forecast(steps=ahead)[2]


# In[49]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
plt.show()

plt.figure(figsize=(12,3))
statsmodels.graphics.tsaplots.plot_acf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(121))
statsmodels.graphics.tsaplots.plot_pacf(y, lags=50, zero=True, use_vlines=True, alpha=0.05, ax=plt.subplot(122))
plt.show()


# In[47]:


ax=pd.DataFrame(y).plot(figsize=(12,5))
forecast_index=[i for i in range(pd.DataFrame(y).index.max()+1, pd.DataFrame(y).index.max()+ahead+1)]
pd.DataFrame(pred_ts_point, index=forecast_index).plot(label='forecast', ax=ax)
ax.fill_between(pd.DataFrame(pred_ts_interval, index=forecast_index).index,
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,0],
                pd.DataFrame(pred_ts_interval, index=forecast_index).iloc[:,1], color='k', alpha=0.15 )
plt.legend(['observed', 'forecast'])


# In[48]:


display(fit.summary2())


# In[ ]:




