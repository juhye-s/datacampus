#!/usr/bin/env python
# coding: utf-8

# ### DBSCAN
# #### iris 분류

# In[1]:


import pandas as pd
from sklearn import datasets
iris=datasets.load_iris()


# In[2]:


iris.feature_names


# In[3]:


iris_df=pd.DataFrame(iris.data, columns=["sepal length", "sepal width","petal length",
                                        "petal width"])
iris_df["labels"]=pd.DataFrame(iris.target)
iris_df.head()


# In[4]:


feature=iris_df[["sepal length", "sepal width","petal length",
                                        "petal width"]]
feature.head()


# In[5]:


from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


model=DBSCAN(eps=0.5, min_samples=5)
clst=model.fit_predict(feature)


# In[7]:


clst


# In[8]:


model.labels_


# In[9]:


predict=pd.DataFrame(clst)
predict.columns=['predict']


# In[10]:


iris_pred=feature.join(predict)
iris_pred


# - #### 시각화

# In[11]:


sns.pairplot(iris_pred, hue='predict')
plt.show()


# In[12]:


sns.pairplot(iris_df,hue='labels')
plt.show


# In[13]:


iris_df["predict"]=predict
iris_df.head()


# In[14]:


iris_result=iris_df.groupby(["labels", "predict"])["sepal length"].count()
iris_result


# ### Kmeans와 비교 

# In[15]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters=3, n_jobs=4, random_state=21)
km.fit(feature)


# In[16]:


predict2=pd.DataFrame(km.labels_)
predict2.columns=['predict2']


# In[17]:


iris_pred2=feature.join(predict2)
iris_pred2


# In[18]:


sns.pairplot(iris_pred2, hue='predict2')
plt.show()


# In[ ]:




