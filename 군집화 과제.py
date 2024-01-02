#!/usr/bin/env python
# coding: utf-8

# ### hierarchical
# #### 와인데이터 군집화

# In[10]:


import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[11]:


wine=datasets.load_wine()


# In[29]:


feature=pd.DataFrame(wine.data)
feature.head()


# In[13]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10,7))
plt.title("wine Dendograms")
dend=shc.dendrogram(shc.linkage(feature))


# In[14]:


from sklearn.cluster import AgglomerativeClustering

cluster=AgglomerativeClustering(n_clusters=3)
cluster.fit_predict(feature)


# In[15]:


label=pd.DataFrame(cluster.labels_)
label.columns=['label']


# In[16]:


wine_pred=feature.join(label)
wine_pred


# In[ ]:


import seaborn as sns

sns.pairplot(wine_pred, hue='label')
plt.show()

