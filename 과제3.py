#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
train=pd.read_csv("train.csv")
train


# In[4]:


#30대 여성이면서 1등급에 탄 사람을 선택(추출)하시오.
cls=train['Pclass']==1
age=(train['Age']>=30)&(train['Age']<40)
train[cls&age]


# In[10]:


#Age의 NaN 값을 다음과 같이 처리하시오
#생존자는 생존자 나이의 평균으로 대체
#사망자는 사망자 나이의 평균으로 대체
survived1=train['Survived']==1
survived0=train['Survived']==0
sur=train[survived1]
nsur=train[survived0]
train['Age'].fillna(sur['Age'].mean())
train['Age'].fillna(nsur['Age'].mean())


# In[11]:


#성별에 따른 생존률을 구하시오
train.groupby(['Sex']).mean()['Survived']


# In[ ]:




