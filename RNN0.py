#!/usr/bin/env python
# coding: utf-8

# ### RNN

# #### 라이브러리 불러오기

# In[3]:


pip install tensorflow


# In[6]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


# 모델 생성(1)

# In[7]:


model=Sequential()
model.add(layers.SimpleRNN(3, input_shape=(2,10)))
model.summary()


# In[8]:


model=Sequential()
model.add(layers.SimpleRNN(3,batch_input_shape=(8,2,10)))
model.summary()


# In[9]:


model=Sequential()
model.add(layers.SimpleRNN(3,batch_input_shape=(8,2,10), return_sequences=True))
model.summary()


# In[ ]:




