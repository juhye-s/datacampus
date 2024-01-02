#!/usr/bin/env python
# coding: utf-8

# ### 군집분석

# In[23]:


import pandas as pd
import numpy as np

from konlpy.tag import Okt
from konlpy.tag import Hannanum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


# In[24]:


#okt=Okt()
han=Hannanum()
df=pd.read_csv('clustering.csv')
df.head()


# In[26]:


doc=[]

for i in df['기사내용']:
    doc.append(han.nouns(i))
    
for i in range(len(doc)):
    doc[i]=' '.join(doc[i])

doc[0]


# In[27]:


cv=CountVectorizer()
X=cv.fit_transform(doc)


# In[28]:


df_cv=pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
df_cv


# In[29]:


kmeans=KMeans(n_clusters=3).fit(df_cv)
kmeans.labels_


# In[30]:


len(cv.get_feature_names())


# In[31]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[32]:


pca=PCA(n_components=2)
PC=pca.fit_transform(df_cv)


# In[33]:


PC_df=pd.DataFrame(data=PC, columns=['pc1','pc2'])
PC_df.index=df['검색어']


# In[34]:


plt.scatter(PC_df.iloc[kmeans.labels_==0,0],PC_df.iloc[kmeans.labels_==0,1],
           s=10, c='red', label='cluster1')
plt.scatter(PC_df.iloc[kmeans.labels_==1,0],PC_df.iloc[kmeans.labels_==1,1],
           s=10, c='blue', label='cluster2')
plt.scatter(PC_df.iloc[kmeans.labels_==2,0],PC_df.iloc[kmeans.labels_==2,1],
            s=10, c='green', label='cluster3')

plt.legend()
plt.show()


# In[ ]:





# ### 계층 분할

# In[17]:


from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc


# In[19]:


cluster=AgglomerativeClustering(n_clusters=3, linkage='ward')
cluster.fit_predict(df_cv)


# In[20]:


plt.figure()
plt.title("Customer Dendrograms")
dend=shc.dendrogram(shc.linkage(df_cv, method='ward'))


# In[21]:


print(doc[5])


# In[ ]:


print(doc[11])

