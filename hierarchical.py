#!/usr/bin/env python
# coding: utf-8

# ### hierarchical 계층적 군집분석 

# - from scipy.cluster.hierarchy import dendrogram, linkage

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[5]:


X=np.array([[5,3],[10,30],[15,12],[24,10],[10,15],[85,70],
           [60,78],[70,55],[80,91],[90,35]])


# In[6]:


labels = range(1,11)
plt.figure(figsize=(10,7))

plt.scatter(X[:,0],X[:,1])

for label, x, y in zip(labels, X[:,0],X[:,1]):
    plt.annotate(label,
                xy=(x,y), xytext=(-3,3), textcoords='offset points', ha='right',va='bottom')

plt.show()


# In[9]:


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked=linkage(X,'single')


# In[12]:


labelList=range(1,11)

plt.figure(figsize=(10,7))
dendrogram(linked, labels=labelList)
plt.show


# In[ ]:





# - #### iris 분류

# - from sklearn.cluster import AgglomerativeClustering

# In[14]:


import pandas as pd
from sklearn import datasets


# In[15]:


iris=datasets.load_iris()


# In[16]:


feature=pd.DataFrame(iris.data, columns=["sepal length","sepal width","petal length","petal width"])
feature.head()


# In[18]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10,7))
plt.title("iris Dendograms")
dend=shc.dendrogram(shc.linkage(feature))


# In[19]:


from sklearn.cluster import AgglomerativeClustering


# In[20]:


cluster=AgglomerativeClustering(n_clusters=3)
cluster.fit_predict(feature)


# In[21]:


label=pd.DataFrame(cluster.labels_)
label.columns=['label']


# In[22]:


iris_pred=feature.join(label)
iris_pred


# In[23]:


import seaborn as sns


# In[24]:


sns.pairplot(iris_pred, hue='label')
plt.show()


# In[ ]:




