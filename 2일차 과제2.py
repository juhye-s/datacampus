#!/usr/bin/env python
# coding: utf-8

# ## <연습문제>

# In[4]:


import numpy as np


# ### ndarray 생성 _ 연습문제

# In[11]:


#로또 번호 자동 생성기를 함수를 이용해 만드시오.
x=np.arange(1,46,1)
number=np.random.choice(x,size=(6,),replace=False)
number.sort()
print(number)


# ### 논리값을 이용한 인덱싱

# In[28]:


# x>30 인 원소 필터링
x=np.random.randint(1,100,size=10)
print(x)
over30=x>30
print(x[over30])


# In[30]:


#짝수이고 x<50 인 원소 필터링
even_under50=(x%2==0)&(x<50)
x[even_under50]


# In[31]:


#X<50 이거나 X>80인 원소 필터링
number=(x<50)|(x>80)
x[number]


# ### redwine 연습문제

# In[32]:


redwine=np.loadtxt(fname="wine_red.csv",delimiter=";", skiprows=1)
print(redwine)


# In[35]:


#redwine의 변수 (열)별 평균값을 구하시오
np.mean(redwine, axis=0)


# In[39]:


#redwine의 alcohol 평균값만 구하시오
np.mean(redwine[:,1])


# In[43]:


#alcohol이 9.5 이상인 와인은 몇개 인지 구하시오
alcohol_list=redwine[:,1]
ac95=(alcohol_list>9.5)
len(alcohol_list[ac95])


# In[44]:


#alcohol이 9.5 이상인 와인의 평균 alcohol값은 얼마인지 구하시오.
np.mean(alcohol_list[ac95])


# In[ ]:




