#!/usr/bin/env python
# coding: utf-8

# ### MA

# In[1]:


import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[4]:


np.r_[1]


# In[5]:


np.random.seed(123)
ar_params=np.array([])
ma_params=np.array([0,9])
ar,ma=np.r_[1, -ar_params], np.r_[1, ma_params]

y=sm.tsa.ArmaProcess(ar, ma).generate_sample(500,burnin=50)


# In[6]:


plt.figure(figsize=(10,4))
plt.plot(y,'o-')
plt.tight_layout()
plt.show()


# In[9]:


plt.figure(figsize=(10,6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar,ma).acf(lags=10))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar,ma).acf(lags=10))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function")

sm.graphics.tsa.plot_acf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling autocorrelation function")
plt.tight_layout()
plt.show()


# In[10]:


np.random.seed(123)
ar_params=np.array([])
ma_params=np.array([-1,0,6]) #y=et-1*et-1+0.6*et-2
ar, ma= np.r_[1, -ar_params], np.r_[1, ma_params]

y=sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)


# In[14]:


plt.figure(figsize=(10,4))
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()


# In[13]:


plt.figure(figsize=(10,6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar,ma).acf(lags=10))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar,ma).acf(lags=10))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function")

sm.graphics.tsa.plot_acf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling autocorrelation function")
plt.tight_layout()
plt.show()


# ### AR

# In[15]:


import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[16]:


np.random.seed(123)
ar_params=np.array([0.9])
ma_params=np.array([])
ar, ma= np.r_[1, -ar_params], np.r_[1, ma_params]

y=sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)


# In[17]:


plt.figure(figsize=(10,4))
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()


# In[19]:


plt.figure(figsize=(10,6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar,ma).acf(lags=10))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar,ma).pacf(lags=10))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function")

sm.graphics.tsa.plot_pacf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling autocorrelation function")
plt.tight_layout()
plt.show()


# In[20]:


#AR2

np.random.seed(123)
ar_params=np.array([0.5,0.25])
ma_params=np.array([])
ar, ma= np.r_[1, -ar_params], np.r_[1, ma_params]

y=sm.tsa.ArmaProcess(ar, ma).generate_sample(500, burnin=50)


# In[21]:


plt.figure(figsize=(10,4))
plt.plot(y, 'o-')
plt.tight_layout()
plt.show()


# In[22]:


plt.figure(figsize=(10,6))
plt.subplot(311)
plt.stem(sm.tsa.ArmaProcess(ar,ma).acf(lags=10))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function")

plt.subplot(312)
plt.stem(sm.tsa.ArmaProcess(ar,ma).pacf(lags=10))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Theoretical autocorrelation function")

sm.graphics.tsa.plot_pacf(y, lags=10, ax=plt.subplot(313))
plt.xlim(-1,11)
plt.ylim(-1.1, 1.1)
plt.title("Sampling autocorrelation function")
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




