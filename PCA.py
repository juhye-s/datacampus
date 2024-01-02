#!/usr/bin/env python
# coding: utf-8

# ### PCA
# #### iris 

# - from sklearn.decomposition import PCA

# In[2]:


import pandas as pd
import numpy as np

from sklearn import datasets


# In[3]:


iris= datasets.load_iris()
x=iris.data[:,[0,2]]
y=iris.target


# In[4]:


print(x.shape, y.shape)


# In[6]:


feature_name=[iris.feature_names[0],iris.feature_names[2]]
x_data=pd.DataFrame(x, columns=feature_name)
x_data.head()


# In[7]:


y_data=pd.DataFrame(y, columns=["target"])
y_data.head()


# In[8]:


from sklearn.decomposition import PCA


# In[9]:


pca=PCA(n_components=2)
pca.fit(x_data)


# In[10]:


pca.explained_variance_


# In[11]:


pca.components_


# In[12]:


PCscore=pca.transform(x_data)
PCscore[0:5]


# In[13]:


x_data[0:5]


# In[14]:


eigens_v=pca.components_.transpose()
print(eigens_v)


# In[16]:


mX=np.matrix(x)

for i in range(x.shape[1]):
    mX[:,i]=mX[:,i]-np.mean(x[:,i])
    
mX_df=pd.DataFrame(mX)


# In[17]:


(mX*eigens_v)[0:5]


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.scatter(PCscore[:,0],PCscore[:,1])
plt.show


# In[20]:


plt.scatter(mX_df[0],mX_df[1])
origin=[0],[0]
plt.quiver((0,0),(0,0), eigens_v[0,:],eigens_v[1,:], color=['r','b'], scale=3)
plt.show()


# In[21]:


plt.scatter(mX_df[0],mX_df[1],c=y)
origin=[0],[0]
plt.quiver((0,0),(0,0), eigens_v[0,:],eigens_v[1,:], color=['r','b'], scale=3)
plt.show()


# ######## [참고] 혹시 PCA 축을 그릴 때 에러가 발생하면 origin 부분을 아래처럼 바꿔주세요. 
# 
# - plt.quiver(*origin     -->      plt.quiver((0,0),(0,0)  

# - #### 회귀분석

# - from sklearn.linear_model import LogisticRegression
# - from sklearn.metrics import confusion_matrix

# In[22]:


x2=iris.data
pca2=PCA(n_components=4)
pca2.fit(x2)


# In[23]:


pca2.explained_variance_


# In[24]:


PCscore2=pca2.transform(x2)[:,0:2]


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[31]:


clf=LogisticRegression(solver="sag", multi_class="multinomial",max_iter=10000).fit(x2,y)


# In[32]:


clf2=LogisticRegression(solver="sag", multi_class="multinomial").fit(PCscore2,y)


# In[36]:


y_pred2=clf2.predict(PCscore2)


# In[37]:


y_pred=clf.predict(x2)


# In[38]:


confusion_matrix(y,y_pred)


# In[39]:


confusion_matrix(y,y_pred2)


# In[ ]:




