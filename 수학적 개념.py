#!/usr/bin/env python
# coding: utf-8

# ## 통계

# - #### 평균 확인

# In[1]:


import numpy as np
a=[70,91,69,78,82]
avg=np.mean(a)

print(avg)


# - #### 분산과 표준편차의 차이 확인

# In[6]:


import numpy as np
a=[173,181,168,175,179]
b=[1.73,1.81, 1.68, 1.75, 1.79]

print(np.var(a))
print(np.var(b))

print(np.std(a))
print(np.std(b))

def standardize(x):
    return(x-np.mean(x))/np.std(x)
a=[173,181,168,175,179]
b=[1.73,1.81, 1.68, 1.75, 1.79]
print(standardize(a))
print(standardize(b))


# - #### 정규분포난수를 생성하고 평균과 표준편차 계산

# In[9]:


import numpy as np
n=np.random.randn(1000)
print(np.mean(n))
print(np.std(n))


# ## 미분

# - #### 2차함수계산

# In[11]:


def f(x):
    return x**2+1

print(f(-2))
print(f(1))


# - #### 미분

# In[12]:


def df(a,h):
    return (f(a+h)-f(a))/h

for h in[1,1e-1,1e-2,1e-3]:
    print([h, df(0,h), df(1,h)])


# ## 행렬

# - #### 행렬의 합과 차 계산

# In[14]:


import numpy as np
A=np.array([[1,2,3],[3,4,5]])
B=np.array([[3,4,5],[4,5,6]])

print(A+B)
print(A-B)


# - #### 행렬의 곱 계산

# In[16]:


A=np.array([[1,2,3],[3,4,5]])
B=np.array([[3,4],[4,5],[5,6]])

print(A.dot(B))
print(A@B)


# - #### 행렬의 역행렬

# In[17]:


A=np.array([[2,5],[1,3]])
print(np.linalg.inv(A))


# - #### 행렬식 계산

# In[18]:


A=np.array([[2,5],[1,3]])
print(np.linalg.det(A))
B=np.array([[1,4,2],[3,-1,-2],[-3,1,3]])
print(np.linalg.det(B))


# In[ ]:




